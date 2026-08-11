# D-TAIA: Domain-Aware Training and Attention-based Inference Architecture

Code for the ECML-PKDD AI4PM workshop paper *"D-TAIA: Domain-Aware LLM
Adaptation for Multi-Task Predictive Process Monitoring."*

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or later |
| PyTorch | 2.0 or later |
| CUDA (optional) | 11.8 or later (CPU fallback available) |
| Git | any |

## Installation

```bash
git clone https://github.com/SvStraten/dtaia.git
cd dtaia

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -e .
```

This installs the `d_taia` package in editable mode along with all
dependencies listed in `pyproject.toml`. If you need a pinned/reproducible
set of exact versions instead, use `pip install -r requirements.txt`.

## HuggingFace Setup

The backbone model `arnir0/Tiny-LLM` downloads automatically from
HuggingFace on first use. You'll need a free HuggingFace account and a
read-scoped access token:

```bash
huggingface-cli login
# paste your token when prompted
```

Model weights (~200 MB) are cached locally after the first download. To
change the cache location, set `hf_cache_dir` in `d_taia/config.py`.

**Do not commit tokens.** If you need a local `.env` for other secrets
(e.g. a Weights & Biases key), it is gitignored by default — verify with
`git check-ignore -v .env` before your first commit.

## Data Preparation

D-TAIA expects **BPI (Business Process Intelligence) event logs** in XES
format, available from the [4TU Research Data portal](https://data.4tu.nl/).
Place downloaded `.xes` files in `data_functions/raw_data/` (created
automatically, gitignored).

## Running the Pipeline

All commands run from the repository root with the virtual environment
active.

### Full pipeline from a raw XES file

```bash
python -m d_taia.pipeline --filepath data_functions/raw_data/BPI_Challenge_2012.xes --dataset bpi2012
```

### Skip data prep (already-processed CSVs exist)

```bash
python -m d_taia.pipeline --dataset bpi2012 --skip-data-prep
```

### Ablations

Ablations are not separate scripts — they're flag combinations on the same
entry point, matching Table 3 in the paper:

```bash
python -m d_taia.pipeline --dataset bpi2015_2 --no-datl
python -m d_taia.pipeline --dataset bpi2015_2 --no-domain-id
python -m d_taia.pipeline --dataset bpi2015_2 --no-faiss
python -m d_taia.pipeline --dataset bpi2015_2 --no-taia
```

### Baselines

Likewise, the two baselines reported in the paper are flag combinations
rather than separate implementations:

```bash
# FT-LLM (Oyamada-style): same TinyLLM backbone, no DATL pretraining,
# no TAIA attention/FFN split, no retrieval — direct regression head only
python -m d_taia.pipeline --dataset bpi2012 --no-datl --no-taia

# MT-RNN: LSTM backbone from scratch, all LLM-specific mechanisms off
python -m d_taia.pipeline --dataset bpi2012 --backbone-lstm --no-datl --no-taia
```

### Experiment sweeps (Figures 3–5)

```bash
python experiments.py backbone --dataset bpi2012
python experiments.py data-pct --dataset bpi2012
python experiments.py prefix-length --dataset bpi2012
```

## Running Tests

```bash
pytest tests/
```

`tests/test_d_taia.py` includes a leakage guard that asserts
`remaining_time` and `accumulated_time` are never present among the
model's input features, and an end-to-end smoke test on a small synthetic
log.

## Project Structure

```
d_taia/
├── d_taia/            # package: config, data, retrieval, heads, backbone,
│                       #   model, metrics, engine, pipeline
├── experiments.py      # sweeps for Figures 3-5, as subcommands
├── scripts/             # SLURM / shell wrappers around the above
├── tests/
└── data_functions/       # raw_data/ (gitignored) + clean_data/ summary
```