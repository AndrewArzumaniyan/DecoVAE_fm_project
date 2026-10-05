# decova-fm

DecoVAE-FM

## Docker

Build image:

```bash
bash docker_build.sh
```

Run and connect to container:

```bash
bash docker_run.sh
docker attach {username}-decovae-fm
```

Then install dependencies in the container shell and activate the environment:

```bash
uv venv --relocatable
uv sync
source .venv/bin/activate
```

## Dataset selection

By default training uses every dataset listed in `./data/manifest.json`. Restrict the subset with Hydra `data.datasets` (case-insensitive name matching):

```bash
# all datasets (default)
python scripts/train.py experiment=base

# Hydra list override
python scripts/train.py experiment=base \
  'data.datasets=[Electricity,London_Smart_Meters,Pedestrian_Counts,Solar,azure_vm_traces_2017,mexico_city_bikes]'

# comma-separated string override (also supported)
python scripts/train.py experiment=base \
  'data.datasets=Electricity,Solar'
```

## MLflow experiment tracking

Training and evaluation log to a local SQLite store at `./mlflow.db` by default
(artifacts remain under `./mlruns`).

Train with tracking (enabled by default):

```bash
python scripts/train.py experiment=base
python scripts/train.py experiment=debug mlflow.enabled=true
```

Evaluate and log metrics to the same experiment:

```bash
python scripts/evaluate.py eval.checkpoint=path/to/epoch_0.ckpt eval.split=val
```

View runs in the MLflow UI (uses `sqlite:///mlflow.db` by default):

```bash
mlflow ui
```

Disable tracking for a run:

```bash
python scripts/train.py mlflow.enabled=false
```