# Local Setup Guide

## Requirements

- Python 3.10
- `protoc` (Protocol Buffer compiler) — [install instructions](https://grpc.io/docs/protoc-installation/)
- Waymo Open Dataset TFRecord files — must be downloaded separately (see below)
- macOS or Linux recommended

---

## 1. Clone the repo

```bash
git clone https://github.com/rafaelmaranon/waymo-validation-lab.git
cd waymo-validation-lab
```

---

## 2. Create virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Compile protobuf classes (once only)

The `.proto` files are included in the repo. You must compile them locally.

```bash
protoc --proto_path=proto --python_out=proto \
  proto/waymo_open_dataset/protos/map.proto \
  proto/waymo_open_dataset/protos/scenario.proto
```

This generates `map_pb2.py` and `scenario_pb2.py` inside `proto/waymo_open_dataset/protos/`.

> ℹ️ No TensorFlow, no Waymo pip package required — only the standard `protobuf` Python package.

---

## 4. Download Waymo data

The raw dataset is **not included** in this repo. You must download it yourself.

1. Register at: https://waymo.com/open/
2. Download the **Waymo Open Motion Dataset v1.3.1** validation split
3. Place the TFRecord files at:

```
~/datasets/waymo/raw/
```

Expected structure:
```
~/datasets/waymo/raw/
└── uncompressed_scenario_validation_validation.tfrecord-*.gz
```

> ⚠️ Do NOT place raw data inside the repo folder. It will not be gitignored there.

---

## 5. Configure scenario limit

The number of scenarios to parse is controlled by a single variable at the top of `scripts/waymo_real_parser.py`:

```python
MAX_SCENARIOS = 250  # Set to None to parse all available scenarios
```

- Default: `250` scenarios
- To parse all: set to `None`
- To test quickly: set to `10` or `20`

---

## 6. Run the pipeline

Run scripts in this order:

```bash
python scripts/waymo_real_parser.py          # Parse TFRecords → silver parquet
python scripts/compute_basic_metrics.py      # Basic scenario stats
python scripts/compute_interaction_metrics.py # SDC interaction metrics
python scripts/compute_risk_metrics.py        # TTC / closing speed / risk score
python scripts/compute_comfort_metrics.py     # Acceleration / jerk / comfort score
python scripts/generate_preview_gifs.py       # Animated GIF previews per scenario
```

Expected outputs under `data/`:
```
data/silver/
    scenarios.parquet
    tracks.parquet
    states.parquet
data/gold/
    scenario_metrics.parquet
    interaction_metrics.parquet
    risk_metrics.parquet
    comfort_metrics.parquet
data/previews/
    <scenario_id>.gif   (one per scenario)
```

---

## 7. Launch the app

```bash
streamlit run scripts/app.py
```

Opens at: http://localhost:8501

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'scenario_pb2'` | Protos not compiled | Run the `protoc` command in step 3 |
| `FileNotFoundError: scenarios.parquet` | Parser not run yet | Run `waymo_real_parser.py` first |
| `No TFRecord files found` | Wrong raw data path | Check `~/datasets/waymo/raw/` exists and has `.tfrecord` files |
| `ModuleNotFoundError: No module named 'plotly'` | Missing dependency | Run `pip install -r requirements.txt` |
| App loads but shows no GIFs | `generate_preview_gifs.py` not run | Run the gif generation step |
