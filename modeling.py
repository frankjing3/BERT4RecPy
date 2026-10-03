"""PyTorch port of FeiSun/BERT4Rec's BERT encoder and masked-item head."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BertConfig:
    vocab_size: int = 0
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 256
    hidden_act: str = 'gelu'
    hidden_dropout_prob: float = 0.2
    attention_probs_dropout_prob: float = 0.2
    max_position_embeddings: int = 200
    type_vocab_size: int = 2
    initializer_range: float = 0.02

    @classmethod
    def from_json_file(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        for name in ('vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
                     'intermediate_size', 'max_position_embeddings', 'type_vocab_size'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.hidden_size % self.num_attention_heads:
            raise ValueError('hidden_size must be divisible by num_attention_heads')
        if self.initializer_range <= 0:
            raise ValueError('initializer_range must be positive')
        for rate in (self.hidden_dropout_prob, self.attention_probs_dropout_prob):
            if not 0 <= rate < 1:
                raise ValueError('dropout probabilities must be in [0, 1)')


def activation(name):
    if name == 'gelu':
        # Matches the tanh approximation in the original BERT implementation.
        return nn.GELU(approximate='tanh')
    if name == 'relu':
        return nn.ReLU()
    if name == 'tanh':
        return nn.Tanh()
    if name in ('linear', ''):
        return nn.Identity()
    raise ValueError(f'unsupported activation: {name}')


class BertBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.heads = config.num_attention_heads
        self.head_dim = width // self.heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.attention_output = nn.Linear(width, width)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_norm = nn.LayerNorm(width, eps=1e-12)
        self.intermediate = nn.Linear(width, config.intermediate_size)
        self.activation = activation(config.hidden_act)
        self.output = nn.Linear(config.intermediate_size, width)
        self.output_norm = nn.LayerNorm(width, eps=1e-12)

    def forward(self, hidden, valid):
        batch, length, width = hidden.shape

        def split(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.query(hidden)), split(self.key(hidden)), split(self.value(hidden))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = valid[:, None, None, :]
        # Bidirectional key masking, with a finite result even for an empty row.
        weights = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(-1)
        weights = weights.masked_fill(~allowed, 0)
        attended = (self.attention_dropout(weights) @ v).transpose(1, 2).reshape(batch, length, width)
        hidden = self.attention_norm(hidden + self.hidden_dropout(self.attention_output(attended)))
        output = self.output(self.activation(self.intermediate(hidden)))
        return self.output_norm(hidden + self.hidden_dropout(output)) * valid.unsqueeze(-1)


class BERT4Rec(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        width = config.hidden_size
        # As in BERT, the embedding matrix (including special tokens) is trainable
        # and shared with the full-vocabulary output classifier.
        self.item_embeddings = nn.Embedding(config.vocab_size, width)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, width)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, width)
        self.embedding_norm = nn.LayerNorm(width, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([BertBlock(config) for _ in range(config.num_hidden_layers)])
        self.prediction_transform = nn.Sequential(nn.Linear(width, width), activation(config.hidden_act),
                                                  nn.LayerNorm(width, eps=1e-12))
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.apply(self._initialize)

    def _initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = self.config.initializer_range
            nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias)

    @property
    def device(self):
        return self.item_embeddings.weight.device

    def encode(self, input_ids):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.max_position_embeddings:
            raise ValueError('input_ids must be [batch, length] within max_position_embeddings')
        valid = input_ids.ne(0)
        positions = torch.arange(input_ids.shape[1], device=self.device)
        hidden = (self.item_embeddings(input_ids) + self.position_embeddings(positions)
                  + self.type_embeddings(torch.zeros_like(input_ids)))
        hidden = self.dropout(self.embedding_norm(hidden)) * valid.unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, valid)
        return hidden

    def masked_features(self, input_ids, positions):
        hidden = self.encode(input_ids)
        positions = torch.as_tensor(positions, dtype=torch.long, device=self.device)
        selected = hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        return self.prediction_transform(selected)

    def forward(self, input_ids, masked_positions):
        features = self.masked_features(input_ids, masked_positions)
        return F.linear(features, self.item_embeddings.weight, self.output_bias)

    @staticmethod
    def loss(logits, labels, weights):
        labels = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
        weights = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
        if not weights.gt(0).any():
            raise ValueError('batch has no masked targets')
        losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none')
        return (losses * weights.flatten()).sum() / (weights.sum() + 1e-5)

    @torch.no_grad()
    def predict(self, input_ids, positions, candidates):
        """Score candidates at one masked position per sequence, without full softmax."""
        was_training = self.training
        self.eval()
        try:
            features = self.masked_features(input_ids, positions)[:, 0]
            candidates = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
            embeddings = self.item_embeddings(candidates)
            if candidates.ndim == 1:
                return features @ embeddings.T + self.output_bias[candidates]
            return torch.einsum('bd,bcd->bc', features, embeddings) + self.output_bias[candidates]
        finally:
            self.train(was_training)
