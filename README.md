# Waymo Validation Lab

A local-first Waymo scenario analytics pipeline designed for dataset parsing and analytics, not ML training.

## 🏗️ Pipeline Architecture

**Data Layers:**
- **Bronze** = Raw TFRecord files (stored in `~/datasets/waymo/raw/`)
- **Silver** = Normalized tables (parquet)
- **Gold** = Derived metrics (parquet)
- **Exports** = Visualizations and JSON (plots, GIFs)

**Memory-Safe Design:**
- Raw datasets stored outside workspace
- Batch scenario parsing (one at a time)
- Immediate parquet export (no large in-memory accumulation)
- Lightweight stack (pandas, numpy, pyarrow, protobuf — no TensorFlow)

**Migration Path:** Local parquet → BigQuery (same logical schema)

**Current Pilot:** First 10 scenarios from validation dataset

## 🔍 Parser Status

**Current Mode:** `real_waymo_protobuf`

All fields are decoded directly from the official Waymo Scenario protobuf schema.
Zero heuristics. Zero synthetic generation.

**What is truly decoded from protobuf:**
- `scenario_id` — real, e.g. `19a486cd29abd7a7`
- `sdc_track_index` — real
- `tracks[].id` — real
- `tracks[].object_type` — real (VEHICLE, PEDESTRIAN, CYCLIST, OTHER)
- `tracks[].states[].center_x` — real
- `tracks[].states[].center_y` — real
- `tracks[].states[].center_z` — real
- `tracks[].states[].length/width/height` — real
- `tracks[].states[].heading` — real
- `tracks[].states[].velocity_x` — real
- `tracks[].states[].velocity_y` — real
- `tracks[].states[].valid` — real

**What is synthetic:** Nothing. All data is real.

**How it works:**
1. Official Waymo `.proto` files downloaded from [waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset)
2. Compiled to Python classes with `protoc` (stored in `proto/`)
3. TFRecord read with pure Python (no TensorFlow)
4. Each record decoded via `scenario_pb2.Scenario.ParseFromString()`

**Sample real output (first scenario):**
```
scenario_id      : 19a486cd29abd7a7
sdc_track_index  : 10
tracks           : 11
steps            : 91
first center_x   : 8382.083984
first center_y   : 7213.890137
first velocity_y : -19.819336
```

**Blocker for previous parsers:** Missing compiled protobuf classes.
The `.proto` files were always publicly available — the fix was compiling them with `protoc`.

## 🚀 Quick Start

```bash
# Setup environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Compile protos (only needed once)
protoc --proto_path=proto --python_out=proto \
  proto/waymo_open_dataset/protos/map.proto \
  proto/waymo_open_dataset/protos/scenario.proto

# Run pipeline
python scripts/waymo_real_parser.py
python scripts/compute_basic_metrics.py
python scripts/compute_interaction_metrics.py
python scripts/compute_risk_metrics.py
python scripts/validate_outputs.py

# Generate visualizations
python scripts/plot_first_scenario.py
python scripts/animate_first_scenario.py

# Launch dashboard
streamlit run scripts/app.py
```

## 📁 Project Structure

```
waymo-validation-lab/
├── README.md
├── requirements.txt
├── .gitignore
├── proto/                          # Compiled Waymo protobuf classes
│   └── waymo_open_dataset/protos/
│       ├── scenario.proto
│       ├── scenario_pb2.py
│       ├── map.proto
│       └── map_pb2.py
├── data/
│   ├── silver/               # Normalized tables
│   ├── gold/                 # Derived metrics
│   └── exports/              # Plots, GIFs, per-scenario JSON
└── scripts/
    ├── waymo_real_parser.py         # Real protobuf parser (primary)
    ├── decode_one_scenario.py       # Proof-of-work single decode
    ├── compute_basic_metrics.py     # Calculate basic scenario metrics
    ├── compute_interaction_metrics.py # Calculate SDC interaction metrics
    ├── compute_risk_metrics.py      # Calculate risk metrics from TTC/closing speed
    ├── app.py                       # Streamlit validation dashboard
    ├── plot_first_scenario.py       # Generate trajectory plot
    ├── animate_first_scenario.py    # Create animation GIF
    └── validate_outputs.py          # Validate data consistency
```

## 🗃️ Data Schema

### Silver Tables

**scenarios**
- scenario_id, source_file, scenario_index, sdc_track_index
- num_tracks, num_steps, objects_of_interest_count, data_source

**tracks**  
- scenario_id, track_id, track_index, object_type
- is_sdc, states_count

**states**
- scenario_id, track_id, track_index, timestep
- x, y, z, length, width, height, heading
- velocity_x, velocity_y, valid, object_type, is_sdc

### Gold Tables

**scenario_metrics**
- scenario_id, source_file, scenario_index
- num_tracks, num_valid_state_rows
- min_x, max_x, min_y, max_y
- approx_num_moving_tracks, avg_speed_mps, max_speed_mps

**interaction_metrics**
- scenario_id, min_sdc_distance_m, mean_min_sdc_distance_m
- num_close_interactions, num_timesteps_with_close_actor
- closest_actor_type, closest_actor_track_id
- sdc_avg_speed_mps, sdc_max_speed_mps, sdc_distance_traveled_m
- num_unique_close_actors, scenario_interest_score

**risk_metrics**
- scenario_id, risk_score, min_ttc_s, max_closing_speed_mps
- num_ttc_below_3s, num_ttc_below_1_5s
- closest_risk_actor_track_id, closest_risk_actor_type
- min_risk_distance_m, risk_score_components

## ⚠️ Environment Notes

- **No TensorFlow required.** Protobuf classes compiled locally with `protoc`.
- **No Waymo pip package required.** Only standard `protobuf` Python package.
- **Raw data:** TFRecord files stored in `~/datasets/waymo/raw/`
- **Stack:** Python 3.10, pandas, numpy, pyarrow, protobuf, matplotlib

## 🎯 Visual Validation

**Trajectory Plot:** `data/exports/first_scenario_plot.png`
- Real x/y trajectories from decoded protobuf
- Color-coded by object type
- SDC track highlighted

**Animation:** `data/exports/first_scenario.gif`
- 91-frame animation of real actor movements
- 9.1 seconds of real scenario time

## 📊 Real Data Statistics (10 scenarios)

- **Total tracks:** 999 (947 vehicles, 50 pedestrians, 2 cyclists)
- **Total state rows:** 90,909
- **Tracks per scenario:** 11–274 (mean 99.9)
- **Max speed:** 30.09 m/s (~67 mph)
- **Avg speed:** 0.19–11.32 m/s across scenarios

## 🗺️ BigQuery Migration

The parquet tables map directly to BigQuery tables with identical schemas:

```sql
CREATE TABLE `project.dataset.scenarios` (...);
CREATE TABLE `project.dataset.tracks` (...);
CREATE TABLE `project.dataset.states` (...);
CREATE TABLE `project.dataset.scenario_metrics` (...);
```

## 🔮 Future Extensions

- Interaction metrics
- Safety metrics  
- Comfort metrics
- Scenario ranking
- Selective rich-map re-parsing for top scenarios

---

*Last updated: 2026-03-10*
*Version: 0.2.0 — real_waymo_protobuf*
