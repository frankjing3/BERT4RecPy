import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from data import MaskedSequenceDataset, load_data, prepare_data
from modeling import BERT4Rec, BertConfig
from optimization import BertAdam, create_optimizer, learning_rate_at
from util import evaluate, evaluation_candidates


def config():
    return BertConfig(vocab_size=14, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                      num_attention_heads=2, max_position_embeddings=6,
                      hidden_dropout_prob=0, attention_probs_dropout_prob=0)


def tiny_data(directory):
    path = Path(directory) / 'tiny.txt'
    path.write_text(''.join(f'{user} {item}\n' for user, items in
                           {2: [1, 2, 3, 4, 5], 7: [6, 7, 8, 9, 10], 9: [1, 2]}.items()
                           for item in items))
    return prepare_data(path)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.model = BERT4Rec(config())

    def test_attention_is_bidirectional(self):
        original = self.model.encode([[1, 2, 3, 0]])
        changed = self.model.encode([[1, 2, 4, 0]])
        self.assertGreater((original[0, 0] - changed[0, 0]).abs().max().item(), 1e-7)

    def test_padding_does_not_change_real_features(self):
        short = self.model.encode([[1, 2]])
        padded = self.model.encode([[1, 2, 0, 0]])
        torch.testing.assert_close(short, padded[:, :2], atol=1e-6, rtol=1e-5)
        empty = self.model.encode([[0, 0, 0]])
        torch.testing.assert_close(empty, torch.zeros_like(empty))
        (padded.square().sum() + empty.square().sum()).backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_loss_ignores_zero_weight_targets(self):
        logits = torch.randn(2, 3, 14, requires_grad=True)
        labels = torch.tensor([[2, 3, 0], [4, 0, 0]])
        weights = labels.ne(0).float()
        expected = F.cross_entropy(logits[weights.bool()], labels[weights.bool()], reduction='sum') / (3 + 1e-5)
        torch.testing.assert_close(self.model.loss(logits, labels, weights), expected)
        self.model.loss(logits, labels, weights).backward()
        self.assertEqual(logits.grad[~weights.bool()].abs().sum().item(), 0)

    def test_tied_prediction_and_checkpoint(self):
        inputs, positions, candidates = [[1, 12, 0]], [[1]], [2, 3, 5]
        expected = self.model(inputs, positions)[:, 0, candidates]
        torch.testing.assert_close(expected, self.model.predict(inputs, positions, candidates))
        self.assertTrue(self.model.training)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save(self.model.state_dict(), path)
            restored = BERT4Rec(config())
            restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(expected, restored.predict(inputs, positions, [candidates]))

    def test_training_reduces_masked_loss(self):
        optimizer = create_optimizer(self.model, 0.003)
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = self.model.loss(self.model([[1, 12, 3]], [[1]]), [[2]], [[1]])
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            optimizer.step()
        self.assertLess(losses[-1], losses[0])

    def test_optimizer_matches_original_uncorrected_adam(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = BertAdam([parameter], lr=0.01)
        grad = torch.tensor([0.2, -0.3])
        parameter.grad = grad.clone()
        expected = parameter.detach().clone() - 0.01 * (0.1 * grad / ((0.001 * grad.square()).sqrt() + 1e-6) + 0.01 * parameter.detach())
        optimizer.step()
        torch.testing.assert_close(parameter, expected)
        self.assertEqual(learning_rate_at(0, 1, 10, 2), 0)
        self.assertEqual(learning_rate_at(1, 1, 10, 2), 0.5)
        self.assertEqual(learning_rate_at(2, 1, 10, 2), 0.8)
        self.assertEqual(learning_rate_at(10, 1, 10, 2), 0)


class DataTests(unittest.TestCase):
    def test_split_vocab_and_zip_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            data = tiny_data(directory)
            mapping = data['vocab']['item_to_id']
            self.assertEqual(data['train'][2], [mapping[x] for x in (1, 2, 3, 4)])
            self.assertEqual(data['test'][2], mapping[5])
            self.assertNotIn(9, data['test'])
            self.assertEqual(data['vocab']['vocab_size'], 14)
            path = Path(directory) / 'tiny.pt'
            torch.save(data, path)
            self.assertEqual(load_data(path), data)
            with ZipFile(Path(directory) / 'tiny.zip', 'w') as archive:
                archive.write(Path(directory) / 'tiny.txt', arcname='tiny.txt')
            self.assertEqual(load_data(Path(directory) / 'tiny.zip'), data)

    def test_masks_windows_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            data = tiny_data(directory)
        dataset = MaskedSequenceDataset(data, max_seq_length=3, max_predictions_per_seq=2,
                                        masked_lm_prob=1, dupe_factor=2)
        self.assertEqual(len(dataset), len(dataset.windows) * 3)
        for index in range(len(dataset)):
            inputs, positions, labels, weights = dataset[index]
            for a, b in zip(dataset[index], dataset[index]):
                torch.testing.assert_close(a, b)
            selected = weights.bool()
            self.assertTrue(torch.all(inputs[positions[selected]] == data['vocab']['mask_token_id']))
            self.assertTrue(torch.all(labels[selected] > 0))
            user, start, end = dataset.windows[index % len(dataset.windows)]
            original = data['train'][user][start:end]
            self.assertEqual(labels[selected].tolist(), [original[p] for p in positions[selected]])
            if index >= 2 * len(dataset.windows):
                self.assertEqual(weights.sum().item(), 1)
                self.assertEqual(positions[0].item(), len(original) - 1)

    def test_evaluation_candidates_and_final_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            data = tiny_data(directory)
        candidates = evaluation_candidates(data, 2, 100, np.random.default_rng(1))
        self.assertEqual(len(candidates), 6)
        self.assertEqual(len(set(candidates)), len(candidates))
        self.assertFalse(set(candidates[1:]) & set(data['histories'][2]))
        mask = data['vocab']['mask_token_id']

        class PerfectModel:
            def predict(self, inputs, positions, candidates):
                assert inputs[0, positions[0][0]] == mask
                return torch.tensor([[1.0] + [0.0] * (len(candidates) - 1)])

        result = evaluate(PerfectModel(), data, 3)
        self.assertEqual(result['users'], 2)
        self.assertEqual(result['ndcg@10'], 1)
        self.assertEqual(result['mrr'], 1)
        result = evaluate(PerfectModel(), data, 1)
        self.assertEqual(result['hit@1'], 1)
        data['histories'][2] = list(range(1, 11))
        self.assertIsNone(evaluation_candidates(data, 2, 100, np.random.default_rng(1)))

    def test_invalid_data_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            data = tiny_data(directory)
            path = Path(directory) / 'bad.txt'
            path.write_text('1 0\n')
            with self.assertRaises(ValueError):
                prepare_data(path)
        with self.assertRaises(ValueError):
            MaskedSequenceDataset(data, max_seq_length=2, max_predictions_per_seq=3)
        with self.assertRaises(ValueError):
            MaskedSequenceDataset(data, prop_sliding_window=0)


if __name__ == '__main__':
    unittest.main()
