"""Chronological histories and lazy, reproducible masked-item training examples."""
from collections import defaultdict
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import Dataset

from vocab import build_vocab


def _read_pairs(stream):
    histories = defaultdict(list)
    digest = hashlib.sha256()
    for number, line in enumerate(stream, 1):
        digest.update(line.encode('utf-8'))
        if not line.strip():
            continue
        try:
            user, item = map(int, line.split())
            if user < 1 or item < 1:
                raise ValueError
        except ValueError as error:
            raise ValueError(f'line {number}: expected two positive integer IDs') from error
        histories[user].append(item)
    if not histories:
        raise ValueError('dataset contains no interactions')
    return dict(histories), digest.hexdigest()


def prepare_data(path):
    path = Path(path)
    if path.suffix == '.zip':
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist()
                     if name.endswith('.txt') and not name.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError('dataset zip must contain one interaction .txt file')
            with archive.open(names[0]) as raw, io.TextIOWrapper(raw, encoding='utf-8') as stream:
                histories, fingerprint = _read_pairs(stream)
    else:
        with path.open(encoding='utf-8') as stream:
            histories, fingerprint = _read_pairs(stream)
    vocab = build_vocab(histories)
    mapping = vocab['item_to_id']
    encoded = {user: [mapping[item] for item in history] for user, history in histories.items()}
    # gen_data_fin.py merges validation back into training: all but the final
    # interaction are training, except histories of length < 3 which stay whole.
    train = {user: history[:-1] if len(history) >= 3 else history for user, history in encoded.items()}
    test = {user: history[-1] for user, history in encoded.items() if len(history) >= 3}
    return {'format_version': 1, 'fingerprint': fingerprint, 'vocab': vocab,
            'train': train, 'test': test, 'histories': encoded}


def load_data(path):
    path = Path(path)
    if path.suffix != '.pt':
        return prepare_data(path)
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format_version') != 1:
        raise ValueError('unsupported prepared-data format')
    return data


class MaskedSequenceDataset(Dataset):
    def __init__(self, data, max_seq_length=200, max_predictions_per_seq=40,
                 masked_lm_prob=0.2, mask_prob=1.0, dupe_factor=10,
                 prop_sliding_window=0.5, seed=12345):
        if max_seq_length < 1 or not 1 <= max_predictions_per_seq <= max_seq_length:
            raise ValueError('require 1 <= max_predictions_per_seq <= max_seq_length')
        if not 0 < masked_lm_prob <= 1 or not 0 <= mask_prob <= 1 or dupe_factor < 0:
            raise ValueError('invalid masking probability or dupe_factor')
        if prop_sliding_window != -1 and not 0 < prop_sliding_window <= 1:
            raise ValueError('prop_sliding_window must be -1 or in (0, 1]')
        self.data = data
        self.maxlen, self.max_predictions = max_seq_length, max_predictions_per_seq
        self.masked_lm_prob, self.mask_prob = masked_lm_prob, mask_prob
        self.dupe_factor, self.seed = dupe_factor, seed
        stride = max_seq_length if prop_sliding_window == -1 else max(1, int(prop_sliding_window * max_seq_length))
        self.windows = []
        for user, sequence in data['train'].items():
            starts = [0] if len(sequence) <= self.maxlen else sorted(set([0] + list(range(len(sequence) - self.maxlen, 0, -stride))))
            self.windows.extend((user, start, min(start + self.maxlen, len(sequence))) for start in starts)
        if not self.windows:
            raise ValueError('no training sequences')

    def __len__(self):
        # Original: dupe_factor random masks plus one force-last example/window.
        return len(self.windows) * (self.dupe_factor + 1)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        replica, window = divmod(index, len(self.windows))
        user, start, end = self.windows[window]
        sequence = self.data['train'][user][start:end]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        if replica == self.dupe_factor:
            positions = np.array([len(sequence) - 1])
        else:
            count = min(self.max_predictions, max(1, round(len(sequence) * self.masked_lm_prob)))
            positions = np.sort(rng.choice(len(sequence), size=count, replace=False))
        input_ids = np.zeros(self.maxlen, dtype=np.int64)
        input_ids[:len(sequence)] = sequence
        labels = np.zeros(self.max_predictions, dtype=np.int64)
        masked_positions = np.zeros(self.max_predictions, dtype=np.int64)
        weights = np.zeros(self.max_predictions, dtype=np.float32)
        for offset, position in enumerate(positions):
            labels[offset], masked_positions[offset], weights[offset] = sequence[position], position, 1.0
            if replica == self.dupe_factor or rng.random() < self.mask_prob:
                input_ids[position] = self.data['vocab']['mask_token_id']
            elif rng.random() >= 0.5:
                input_ids[position] = rng.integers(1, self.data['vocab']['item_count'] + 1)
        return tuple(torch.from_numpy(value) for value in (input_ids, masked_positions, labels, weights))

    def sample_batch(self, batch_size, rng):
        indices = rng.integers(0, len(self), size=batch_size)
        examples = [self[int(index)] for index in indices]
        return tuple(torch.stack(values) for values in zip(*examples))
