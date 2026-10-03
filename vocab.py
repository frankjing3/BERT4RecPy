"""Deterministic frequency-ranked item IDs and BERT special tokens."""
from collections import Counter


def build_vocab(histories):
    counts = Counter(item for history in histories.values() for item in history)
    # Stable tie breaking replaces the source's unordered special-token set.
    items = sorted(counts, key=lambda item: (-counts[item], item))
    item_to_id = {item: index + 1 for index, item in enumerate(items)}
    count = len(items)
    return {'item_to_id': item_to_id, 'id_to_item': [0] + items,
            'counts': [0] + [counts[item] for item in items],
            'item_count': count, 'pad_token_id': count + 1,
            'mask_token_id': count + 2, 'unused_token_id': count + 3,
            'vocab_size': count + 4}
