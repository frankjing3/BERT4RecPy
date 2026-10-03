#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run.py --dataset_name ml-1m "$@"
