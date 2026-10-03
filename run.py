"""Train, resume, or evaluate BERT4Rec using PyTorch on CPU or CUDA."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm import trange

from data import MaskedSequenceDataset, load_data
from modeling import BERT4Rec, BertConfig
from optimization import create_optimizer, learning_rate_at
from util import evaluate

ROOT = Path(__file__).resolve().parent
PRESETS = {
    'ml-1m': (200, 40, 0.2, 0.5, 64, 400000),
    'ml-20m': (200, 20, 0.2, 0.5, 64, 400000),
    'beauty': (50, 30, 0.6, 0.1, 64, 400000),
    'steam': (50, 20, 0.4, 0.5, 256, 800000),
}
DATA_OPTIONS = ('max_seq_length', 'max_predictions_per_seq', 'masked_lm_prob',
                'mask_prob', 'dupe_factor', 'prop_sliding_window', 'seed')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_name', '--dataset', dest='dataset_name', required=True)
    parser.add_argument('--data_dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--data_file', type=Path, help='raw .txt/.zip or prepared .pt histories')
    parser.add_argument('--output_dir', type=Path, help='default: runs/<dataset_name>')
    parser.add_argument('--bert_config_file', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--inference_only', action='store_true')
    for name in ('max_seq_length', 'max_predictions_per_seq', 'dupe_factor', 'batch_size',
                 'num_train_steps', 'num_warmup_steps', 'seed'):
        parser.add_argument(f'--{name}', type=int)
    for name in ('masked_lm_prob', 'mask_prob', 'prop_sliding_window', 'learning_rate'):
        parser.add_argument(f'--{name}', type=float)
    parser.add_argument('--save_checkpoints_steps', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--eval_users', type=int, default=10000, help='0 evaluates all eligible users')
    parser.add_argument('--eval_negatives', type=int, default=100)
    parser.add_argument('--uniform_negatives', action='store_true', help='default: popularity-weighted negatives')
    parser.add_argument('--max_steps_this_run', type=int, help='stop early while preserving the full LR schedule')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 1:
            parser.error('unsupported checkpoint format')
        if checkpoint['args']['dataset_name'] != args.dataset_name:
            parser.error('--dataset_name differs from the checkpoint')
    if args.inference_only and checkpoint is None:
        parser.error('--inference_only requires --checkpoint')
    maxlen, predictions, mask_rate, stride, width, steps = PRESETS.get(args.dataset_name, (50, 10, 0.2, 0.5, 64, 400000))
    defaults = dict(max_seq_length=maxlen, max_predictions_per_seq=predictions, masked_lm_prob=mask_rate,
                    mask_prob=1.0, prop_sliding_window=stride, dupe_factor=10, batch_size=256,
                    num_train_steps=steps, num_warmup_steps=100, learning_rate=1e-4, seed=12345)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, checkpoint['args'][name] if checkpoint else value)
    for name in ('batch_size', 'num_train_steps', 'save_checkpoints_steps', 'eval_interval', 'eval_negatives'):
        if getattr(args, name) < 1:
            parser.error(f'--{name} must be positive')
    if args.max_steps_this_run is not None and args.max_steps_this_run < 1:
        parser.error('--max_steps_this_run must be positive')
    if args.eval_users < 0 or args.seed < 0 or args.learning_rate <= 0 or not 0 <= args.num_warmup_steps < args.num_train_steps:
        parser.error('require nonnegative eval_users/seed, positive learning_rate, and 0 <= warmup < train steps')
    if checkpoint:
        for name in DATA_OPTIONS + ('batch_size', 'learning_rate', 'num_train_steps', 'num_warmup_steps'):
            if getattr(args, name) != checkpoint['args'][name]:
                parser.error(f'--{name} must match the saved run ({checkpoint["args"][name]})')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = args.data_file or args.data_dir / f'{args.dataset_name}.txt'
    if not source.exists() and args.data_file is None:
        source = source.with_suffix('.zip')
    data = load_data(source)
    if checkpoint and (checkpoint['fingerprint'] != data['fingerprint'] or checkpoint['vocab'] != data['vocab']):
        parser.error('dataset contents or vocabulary differ from the checkpoint')
    if checkpoint:
        config = BertConfig(**checkpoint['config'])
    elif args.bert_config_file:
        config = BertConfig.from_json_file(args.bert_config_file)
    else:
        preset_config = ROOT / 'bert_train' / f'bert_config_{args.dataset_name}_{width}.json'
        config = BertConfig.from_json_file(preset_config) if preset_config.exists() else BertConfig()
    # Source config vocabulary sizes are stale placeholders; derive from actual data.
    config.vocab_size = data['vocab']['vocab_size']
    if checkpoint and args.bert_config_file:
        requested = BertConfig.from_json_file(args.bert_config_file)
        requested.vocab_size = config.vocab_size
        if requested != config:
            parser.error('--bert_config_file differs from the checkpoint')
    if args.max_seq_length > config.max_position_embeddings:
        parser.error('--max_seq_length exceeds the model position embedding table')
    try:
        config.validate()
        dataset = MaskedSequenceDataset(data, **{name: getattr(args, name) for name in DATA_OPTIONS})
    except ValueError as error:
        parser.error(str(error))
    model = BERT4Rec(config).to(args.device)
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    print(f'users={len(data["train"])} items={data["vocab"]["item_count"]} training_examples={len(dataset)} device={args.device}')

    def metrics():
        return evaluate(model, data, args.max_seq_length, args.eval_negatives, args.eval_users,
                        args.seed, popularity=not args.uniform_negatives)

    if args.inference_only:
        print(json.dumps({'test': metrics()}))
        return
    optimizer = create_optimizer(model, args.learning_rate)
    rng = np.random.default_rng(args.seed)
    start = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        rng.bit_generator.state = checkpoint['sampler_rng']
        torch.set_rng_state(checkpoint['torch_rng'])
        if torch.cuda.is_available() and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        start = checkpoint['step']
    end = args.num_train_steps
    if args.max_steps_this_run is not None:
        end = min(end, start + args.max_steps_this_run)
    if start >= end:
        print('The requested training steps are already complete.')
        return
    output = args.output_dir or ROOT / 'runs' / args.dataset_name
    output.mkdir(parents=True, exist_ok=True)
    settings = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    (output / 'args.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    (output / 'bert_config.json').write_text(json.dumps(config.to_dict(), indent=2) + '\n', encoding='utf-8')
    (output / 'vocab.json').write_text(json.dumps(data['vocab']) + '\n', encoding='utf-8')
    started, total_loss, logged_steps = time.monotonic(), 0.0, 0
    with (output / 'log.jsonl').open('a', encoding='utf-8') as log:
        for step in trange(start, end, desc='training'):
            model.train()
            inputs, positions, labels, weights = [tensor.to(args.device) for tensor in dataset.sample_batch(args.batch_size, rng)]
            lr = learning_rate_at(step, args.learning_rate, args.num_train_steps, args.num_warmup_steps)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model(inputs, positions), labels, weights)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item()
            logged_steps += 1
            completed = step + 1
            should_eval = completed % args.eval_interval == 0 or completed == end
            should_save = completed % args.save_checkpoints_steps == 0 or completed == end
            if should_eval or should_save:
                record = {'step': completed, 'loss': total_loss / logged_steps, 'lr': lr,
                          'seconds': time.monotonic() - started}
                if should_eval:
                    record['test'] = metrics()
                print(json.dumps(record))
                log.write(json.dumps(record) + '\n')
                log.flush()
                total_loss, logged_steps = 0.0, 0
            if should_save:
                state = {'format_version': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'step': completed, 'config': config.to_dict(), 'args': settings,
                         'vocab': data['vocab'], 'fingerprint': data['fingerprint'],
                         'sampler_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                         'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
                temporary = output / 'latest.tmp'
                torch.save(state, temporary)
                temporary.replace(output / 'latest.pt')
    print(f'Checkpoint: {output / "latest.pt"}')


if __name__ == '__main__':
    main()
