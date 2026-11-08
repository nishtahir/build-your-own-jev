# Build your own Jev (Decision model)

Companion repo for my blog post [Build your own decision model](https://nishtahir.com/build-your-own-decision-model/)

## Setup

Create and source your virtual environment

```sh
uv venv
uv sync
source .venv/bin/activate
```

## Run

Download and format your dataset

```sh
python scripts/prepare_dataset.py
```

Generate a baseline evaluation

```sh
python scripts/eval-lm-head.py
```

Finetune the model on the train split of the dataset
```sh
python scripts/finetune-lm-head.py
```

Run a calibration pass

```sh
python scripts/calibrate-lm-head.py
```

Use your trained model to make some decisions
```sh
python scripts/predict-lm-head.py
```

