# DIY Jev

## Setup

```sh
uv venv
uv sync
source .venv/bin/activate
```

## Run

```sh
python scripts/prepare_dataset.py      # download and format the dataset
python scripts/eval-lm-head.py         # baseline eval
python scripts/finetune-lm-head.py     # train, saves to checkpoint/
python scripts/calibrate-lm-head.py    # fit temperature, writes temperature.json
python scripts/predict-lm-head.py      # predict on input.json
```

## Lint

```sh
ruff format
ruff check
ty check
```
