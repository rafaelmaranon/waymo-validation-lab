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
python scripts/compute_comfort_metrics.py
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
    ├── compute_comfort_metrics.py   # Calculate comfort metrics from acceleration/jerk
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

**comfort_metrics**
- scenario_id, max_acceleration_mps2, max_deceleration_mps2
- max_jerk_mps3, mean_abs_jerk_mps3, max_heading_rate_radps
- comfort_score, comfort_score_components

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
CREATE TABLE `project.dataset.interaction_metrics` (...);
CREATE TABLE `project.dataset.risk_metrics` (...);
```

## 📊 Metric Calculation Logic

### Risk Score Calculation

```mermaid
graph TD
    A[Scenario States] --> B[Identify SDC Track]
    A --> C[Identify Other Actors]
    
    B --> D[Align States by Timestep]
    C --> D
    
    D --> E[Calculate TTC & Closing Speed]
    E --> F[TTC Component<br/>min(1.0, 5.0 / max(min_ttc_s, 0.1))]
    E --> G[Closing Component<br/>min(1.0, max_closing_speed_mps / 15.0)]
    E --> H[Breach Component<br/>min(1.0, (num_ttc_below_3s + 2 * num_ttc_below_1_5s) / 20.0)]
    
    F --> I[Risk Score<br/>0.5 * ttc + 0.3 * closing + 0.2 * breach]
    G --> I
    H --> I
    
    style I fill:#ffcccc
```

### Complexity Score Calculation

```mermaid
graph TD
    A[Interaction Metrics] --> B[Distance Component<br/>min(1.0, 10.0 / (min_sdc_distance_m + 0.1))]
    A --> C[Interaction Component<br/>min(1.0, num_close_interactions / 50.0)]
    A --> D[Speed Component<br/>min(1.0, sdc_max_speed_mps / 25.0)]
    A --> E[Actor Component<br/>min(1.0, num_unique_close_actors / 10.0)]
    A --> F[Movement Component<br/>min(1.0, sdc_distance_traveled_m / 200.0)]
    
    B --> G[Complexity Score<br/>0.30 * distance + 0.25 * interaction + 0.20 * speed + 0.15 * actor + 0.10 * movement]
    C --> G
    D --> G
    E --> G
    F --> G
    
    style G fill:#cce5ff
```

### Comfort Score Calculation

```mermaid
graph TD
    A[SDC States] --> B[Calculate Speed<br/>sqrt(vx² + vy²)]
    A --> C[Sort by Timestep]
    
    B --> D[Compute Acceleration<br/>dv/dt]
    C --> D
    
    D --> E[Compute Jerk<br/>da/dt]
    D --> F[Compute Heading Rate<br/>wrapped_angle_diff / dt]
    
    E --> G[Acceleration Component<br/>min(1.0, max_acceleration_mps2 / 4.0)]
    E --> H[Deceleration Component<br/>min(1.0, max_deceleration_mps2 / 4.0)]
    E --> I[Jerk Component<br/>min(1.0, max_jerk_mps3 / 10.0)]
    F --> J[Heading Component<br/>min(1.0, max_heading_rate_radps / 0.8)]
    
    G --> K[Comfort Score<br/>0.25*accel + 0.30*decel + 0.30*jerk + 0.15*heading]
    H --> K
    I --> K
    J --> K
    
    style K fill:#ccffcc
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
