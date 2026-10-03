"""Next-item evaluation using a final [MASK] token and sampled ranking."""
import numpy as np


def evaluation_candidates(data, user, count, rng, popularity=True):
    vocab = data['vocab']
    excluded = set(data['histories'][user])
    available = np.array([item for item in range(1, vocab['item_count'] + 1) if item not in excluded], dtype=np.int64)
    if not len(available):
        return None
    probabilities = None
    if popularity:
        probabilities = np.asarray(vocab['counts'], dtype=np.float64)[available]
        probabilities /= probabilities.sum()
    negatives = rng.choice(available, size=min(count, len(available)), replace=False, p=probabilities)
    return np.concatenate(([data['test'][user]], negatives))


def evaluate(model, data, max_seq_length, negatives=100, max_users=10000, seed=12345, popularity=True):
    rng = np.random.default_rng(seed)
    users = sorted(data['test'])
    if max_users and len(users) > max_users:
        users = rng.choice(users, size=max_users, replace=False).tolist()
    metrics = {f'{name}@{k}': 0.0 for k in (1, 5, 10) for name in ('ndcg', 'hit')}
    metrics.update(mrr=0.0, users=0, skipped_users=0, mean_negatives=0.0)
    for user in users:
        candidates = evaluation_candidates(data, user, negatives, rng, popularity)
        if candidates is None:
            metrics['skipped_users'] += 1
            continue
        # Right padding and absolute positions match the original TF examples.
        history = data['train'][user][-(max_seq_length - 1):] if max_seq_length > 1 else []
        sequence = history + [data['vocab']['mask_token_id']]
        input_ids = np.zeros((1, max_seq_length), dtype=np.int64)
        input_ids[0, :len(sequence)] = sequence
        scores = model.predict(input_ids, [[len(sequence) - 1]], candidates)[0].cpu().numpy()
        rank = int(np.argsort(-scores, kind='stable').tolist().index(0))
        for k in (1, 5, 10):
            if rank < k:
                metrics[f'hit@{k}'] += 1
                metrics[f'ndcg@{k}'] += 1 / np.log2(rank + 2)
        metrics['mrr'] += 1 / (rank + 1)
        metrics['mean_negatives'] += len(candidates) - 1
        metrics['users'] += 1
    if metrics['users']:
        for key in metrics.keys() - {'users', 'skipped_users'}:
            metrics[key] /= metrics['users']
    return metrics
