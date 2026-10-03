"""Prepare compact PyTorch histories; masked examples are materialized lazily."""
import argparse
from pathlib import Path

import torch

from data import prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--data_dir', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.data_dir / f'{args.dataset_name}.txt'
    if not source.exists():
        source = source.with_suffix('.zip')
    data = prepare_data(source)
    output = args.output or args.data_dir / f'{args.dataset_name}.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    print(f'Saved {len(data["train"])} users, {data["vocab"]["item_count"]} items to {output}')


if __name__ == '__main__':
    main()
