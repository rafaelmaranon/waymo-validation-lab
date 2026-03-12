# Data Pipeline

## Overview

```
Raw TFRecords  (local, not in repo)
    ↓  waymo_real_parser.py
Silver layer   (scenarios, tracks, states)
    ↓  compute_basic_metrics.py
    ↓  compute_interaction_metrics.py
    ↓  compute_risk_metrics.py
    ↓  compute_comfort_metrics.py
Gold layer     (derived metrics per scenario)
    ↓  generate_preview_gifs.py
data/previews/ (animated GIFs)
    ↓  scripts/app.py
Streamlit dashboard
```

---

## Step 1 — Parse TFRecords → Silver

**Script:** `scripts/waymo_real_parser.py`

**Input:** TFRecord files at `~/datasets/waymo/raw/`

**How it works:**
- Reads each TFRecord file as raw bytes
- Decodes each record using `scenario_pb2.Scenario.ParseFromString()`
- Extracts scenario metadata, actor tracks, and per-timestep states
- Writes three normalized parquet tables

**Outputs:**

| File | Description |
|---|---|
| `data/silver/scenarios.parquet` | One row per scenario: ID, track count, step count |
| `data/silver/tracks.parquet` | One row per actor track: type, SDC flag |
| `data/silver/states.parquet` | One row per actor per timestep: position, velocity, heading |

**Scenario limit:** Controlled by `MAX_SCENARIOS` at top of script (default: 250).

---

## Step 2 — Basic Metrics → Gold

**Script:** `scripts/compute_basic_metrics.py`

**Input:** `data/silver/scenarios.parquet`, `data/silver/states.parquet`

**Computes:** Spatial bounds, moving track count, average/max speed per scenario.

**Output:** `data/gold/scenario_metrics.parquet`

---

## Step 3 — Interaction Metrics → Gold

**Script:** `scripts/compute_interaction_metrics.py`

**Input:** Silver tables

**Computes:**
- Minimum SDC-to-actor distance per scenario
- Number of close interactions (actors within 5m of SDC)
- Number of unique close actors
- SDC speed and distance traveled
- `scenario_interest_score` — composite complexity proxy [0–1]

**Output:** `data/gold/interaction_metrics.parquet`

---

## Step 4 — Risk Metrics → Gold

**Script:** `scripts/compute_risk_metrics.py`

**Input:** Silver tables

**Computes:**
- Time-to-Collision (TTC) for all actor/timestep pairs where actors are approaching
- Maximum closing speed
- TTC breach counts (below 3s and 1.5s thresholds)
- `risk_score` — composite interaction score [0–1]:

```
risk_score = 0.5 * ttc_component
           + 0.3 * closing_component
           + 0.2 * breach_component
```

**Output:** `data/gold/risk_metrics.parquet`

---

## Step 5 — Comfort Metrics → Gold

**Script:** `scripts/compute_comfort_metrics.py`

**Input:** Silver tables (SDC states only)

**Computes:**
- Acceleration, deceleration, jerk from SDC velocity over time
- Heading rate from SDC heading over time
- `comfort_score` — composite ride abruptness [0–1]:

```
comfort_score = 0.25 * accel_component
              + 0.30 * decel_component
              + 0.30 * jerk_component
              + 0.15 * heading_component
```

Higher score = less comfortable (more abrupt motion).

**Output:** `data/gold/comfort_metrics.parquet`

---

## Step 6 — Preview GIFs

**Script:** `scripts/generate_preview_gifs.py`

**Input:** Silver tables

**Computes:** Animated bird's-eye-view GIF per scenario showing actor trajectories over time.

**Output:** `data/previews/<scenario_id>.gif`

These GIFs are used directly by the Streamlit app for the scenario preview section.

---

## Step 7 — Streamlit App

**Script:** `scripts/app.py`

**Input:** All gold parquet tables + `data/previews/*.gif` + `data/diagrams/*.png`

**Renders:**
- Hero header with project info and links
- Top scenario GIF previews (highest risk_score first)
- Metric summary cards with histograms (Interaction, Complexity, Comfort)
- Interactive Plotly scatter map (Interaction Score vs Complexity Score)
- Methodology expander with pipeline diagram

---

## Validation

**Script:** `scripts/validate_outputs.py`

Run after the pipeline to check row counts, null rates, and score distributions are within expected ranges.

---

## Notes

- All computations are deterministic given the same input TFRecords
- Parquet files are local only — not committed to the repo
- The app reads all parquets at startup; missing tables show graceful error messages
- `data/diagrams/` contains static PNG files used by the methodology expander — these are committed to the repo
