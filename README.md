# BERT4RecPy

Python 3 / PyTorch conversion of [FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec), for **BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer** (Sun et al., CIKM 2019).

The model, masked-data generation, optimization, training and ranking evaluation run without TensorFlow. The original datasets, all twelve model JSON configurations, and Apache 2.0 license are included. See `NOTICE` for source attribution.

## Install

Use Python 3.10 or later. From this project directory:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

CPU and CUDA are supported; the default device is CUDA when available. Install a PyTorch distribution appropriate for your GPU environment if needed, or pass `--device cpu`.

## Train

```bash
python run.py --dataset_name ml-1m
# Equivalent convenience entry point:
bash run_ml-1m.sh
```

Dataset names: `ml-1m`, `ml-20m`, `beauty`, `steam` (case-sensitive). The MovieLens 20M ZIP can be read directly without extraction.

The dataset-specific defaults follow the original shell scripts:

| Dataset | Sequence length | Mask fraction | Maximum predictions | Window stride fraction | Hidden width | Training steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ml-1m | 200 | 0.2 | 40 | 0.5 | 64 | 400,000 |
| ml-20m | 200 | 0.2 | 20 | 0.5 | 64 | 400,000 |
| beauty | 50 | 0.6 | 30 | 0.1 | 64 | 400,000 |
| steam | 50 | 0.4 | 20 | 0.5 | 256 | 800,000 |

Shared defaults: batch size 256, learning rate 1e-4, 100 warmup steps, mask replacement probability 1.0, ten random-mask replicas plus one last-item-mask replica per window. All presets use two transformer layers and two attention heads. A config file can select other supplied widths (64/128/256):

```bash
python run.py --dataset_name beauty --bert_config_file bert_train/bert_config_beauty_128.json
```

A short CPU check (not a recommendation-quality benchmark):

```bash
python run.py --dataset_name beauty --device cpu --batch_size 4 --num_train_steps 4 --num_warmup_steps 1 --eval_users 20 --output_dir runs/smoke
```

Training logs and checkpoints go to `runs/<dataset_name>/` unless `--output_dir` is supplied. Outputs: `latest.pt`, `args.json`, `bert_config.json`, `vocab.json`, and `log.jsonl`. Reusing a directory appends logs and replaces its checkpoint. Model vocabulary size is derived from the data, replacing the placeholder value in the original JSON files.

## Prepare data, resume, and evaluate

Training accepts raw chronological `user_id item_id` pairs (positive integers), or an optional compact prepared file:

```bash
python gen_data_fin.py --dataset_name beauty --output data/beauty.pt
python run.py --dataset_name beauty --data_file data/beauty.pt
```

`gen_data.py` is an alias for the same preparation command. Prepared files contain histories, the vocabulary, and a dataset fingerprint. Masked windows and replicas are generated lazily and deterministically, avoiding large TFRecord files. Training samples those replicas with replacement. `--data_file` also supports a custom `.txt` or `.zip` file; interactions must already be ordered chronologically within each user.

Resume an interrupted run:

```bash
python run.py --dataset_name beauty --checkpoint runs/beauty/latest.pt
```

For a deliberate partial run, use `--max_steps_this_run`; this preserves the original total-step learning-rate schedule:

```bash
python run.py --dataset_name beauty --max_steps_this_run 1000
python run.py --dataset_name beauty --checkpoint runs/beauty/latest.pt
```

Checkpoint resume automatically restores model configuration, masking settings, optimizer state, sampling state and Torch random states. The same dataset and training schedule are required. For nondefault data/output locations, pass `--data_file` and `--output_dir` again. An explicit conflicting training option produces an error. Exact reproducibility across different devices or software versions is not guaranteed.

```bash
python run.py --dataset_name beauty --checkpoint runs/beauty/latest.pt --inference_only --eval_users 0
```

Evaluation uses a final `[MASK]` and reports Hit and NDCG at 1/5/10, plus MRR (the source's `ap`, with one relevant item). By default it evaluates up to 10,000 users with 100 distinct popularity-weighted negatives per user, excluding their full interaction history and target. `--eval_users 0` evaluates all users; `--uniform_negatives` selects uniform sampling. A fixed seed makes repeated evaluations comparable. When fewer negatives are available, all available negatives are used; users with none are skipped. Logs include evaluated/skipped counts and mean negative count, so small-catalog metrics can be interpreted correctly.

## What is preserved and what changes

- The original `gen_data_fin.py` folds validation interactions into training. This port does the same: users with at least three events train on all but the final event, and the final event is the test target. Shorter histories remain entirely in training. Test metrics should not be used for hyperparameter selection; there is no separate validation metric in this source protocol.
- Learned item, absolute-position and zero-segment embeddings feed a bidirectional BERT encoder with post-residual LayerNorm, separate attention/hidden dropout, and tanh-approximate GELU. There is no causal mask.
- The masked-item head has a hidden transform, activation and LayerNorm, a classifier tied to item embeddings, and output bias. Loss is weighted full-vocabulary cross entropy over selected positions, with the original denominator epsilon.
- `BertAdam` preserves the source's Adam update **without bias correction**, decoupled weight decay excluding biases/LayerNorm, epsilon 1e-6, gradient clipping at 5, and warmup/linear decay. With warmup enabled, the first update has zero learning rate, as in the source.
- Histories are right padded with zero. Real items are frequency-ranked IDs starting at one; special token ordering is now deterministic. Masking defaults to always replacing selected items with `[MASK]`; with a lower `--mask_prob`, the remaining probability is split between unchanged and random-item replacements.
- Windows, ten random-mask replicas, and an additional last-item-mask replica follow the original design, but random sequences and training batch order are not bit-for-bit identical to TensorFlow. Single-item histories and empty attention rows are handled safely.
- TensorFlow Estimator/session/TFRecord/TPU plumbing is replaced with ordinary PyTorch training and compact `.pt` data. Old TensorFlow checkpoints, pickled vocabularies and TFRecord inputs are not load-compatible. No automatic weight conversion is included.
- Original popularity evaluation uses repeated rejection draws; this implementation samples directly from eligible items without replacement. Uniform negatives are also distinct here. These changes, the user evaluation cap, and safer small-catalog handling mean exact published metric reproduction is not claimed.

## Files and tests

- `modeling.py`: encoder, tied masked-item head, configuration.
- `data.py`, `vocab.py`, `gen_data_fin.py`: histories, vocabulary and masked windows.
- `optimization.py`: original-style BERT Adam and learning-rate schedule.
- `run.py`, `util.py`: training, resume and sampled evaluation.
- `tests/`: masking, bidirectionality, padding/gradients, loss, optimization, checkpoint predictions, data and evaluation checks.

```bash
python -m unittest discover -s tests -v
python run.py --help
```

CPU smoke tests are not full convergence validation. GPU execution and numerical equivalence to the TensorFlow implementation have not been validated.

Please cite the original paper when using this code or data:

```bibtex
@inproceedings{sun2019bert4rec,
  title={BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
  author={Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  year={2019},
  pages={1441--1450}
}
```
