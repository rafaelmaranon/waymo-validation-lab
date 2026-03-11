# System Architect Recommendations

## SECTION 1 — ARCHITECTURE DECISION

**Next improvement: SDC-centric interaction metrics + scenario ranking score.**

The current gold layer (`scenario_metrics.parquet`) contains only aggregate spatial/kinematic statistics (bounding box, speed). It tells you nothing about *relationships between actors*—which is the core value of a scenario analytics tool.

The correct next step is a single new script that reads the silver `states.parquet` and `tracks.parquet`, computes pairwise SDC-to-actor distances at each timestep, and produces a new gold table `interaction_metrics.parquet` with per-scenario interaction metrics and a composite ranking score. This follows the existing Bronze→Silver→Gold pattern, requires no schema changes to silver, adds one file and one gold table, and immediately enables "show me the most interesting scenarios" queries. It is the smallest change that transforms the system from "parser" into "analytics tool."

---

## SECTION 2 — INSTRUCTIONS FOR SWE

**Before starting, commit the current milestone:**

```bash
cd /Users/rafaelmaranon/waymo-validation-lab
git add .
git commit -m "v1.0: real_waymo_protobuf parser — all fields decoded from official Waymo proto"
git tag v1.0_real_waymo_protobuf
```

---

### Step 1 — Create `scripts/compute_interaction_metrics.py`

**File path:** `/Users/rafaelmaranon/waymo-validation-lab/scripts/compute_interaction_metrics.py`

**Purpose:** Read silver tables, compute SDC-centric interaction metrics per scenario, output to gold.

**Imports required:**
```
sys, pathlib.Path, pandas, numpy
```

**Constants:**
```python
CLOSE_THRESHOLD_M = 5.0   # meters — defines "close interaction"
SILVER_DIR = project_root / 'data' / 'silver'
GOLD_DIR   = project_root / 'data' / 'gold'
```

**Functions to implement (in this order):**

---

#### `load_silver() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

- Load `scenarios.parquet`, `tracks.parquet`, `states.parquet` from `SILVER_DIR`
- Print row counts
- Return `(scenarios_df, tracks_df, states_df)`

---

#### `compute_scenario_interactions(scenario_id: str, states_df: pd.DataFrame, tracks_df: pd.DataFrame) -> dict`

This is the core function. For one scenario:

1. Filter `states_df` to this `scenario_id` and only rows where `valid == True`.
2. Identify the SDC track_id from `tracks_df` where `is_sdc == True` and `scenario_id` matches.
3. Split states into `sdc_states` (SDC track) and `other_states` (everything else).
4. For each timestep present in both `sdc_states` and `other_states`:
   - Get SDC position `(x, y)` at that timestep.
   - Get all other actor positions at that timestep.
   - Compute Euclidean distance from SDC to each other actor.
   - Record the minimum distance at this timestep.
   - Count how many other actors are within `CLOSE_THRESHOLD_M`.
5. Across all timesteps, compute:

| Metric | Computation |
|--------|-------------|
| `min_sdc_distance_m` | Minimum of all per-timestep minimum distances |
| `mean_min_sdc_distance_m` | Mean of all per-timestep minimum distances |
| `num_close_interactions` | Total count of (timestep, actor) pairs within threshold |
| `num_timesteps_with_close_actor` | Number of timesteps where at least one actor is within threshold |
| `closest_actor_type` | Object type of the actor that achieved `min_sdc_distance_m` |
| `closest_actor_track_id` | Track ID of that actor |
| `sdc_avg_speed_mps` | Mean of `sqrt(velocity_x² + velocity_y²)` across valid SDC states |
| `sdc_max_speed_mps` | Max of same |
| `sdc_distance_traveled_m` | Sum of Euclidean distances between consecutive valid SDC positions |
| `num_unique_close_actors` | Number of distinct actors that were ever within threshold |

6. Compute a `scenario_interest_score` as a **normalized composite**:

```python
score = (
    0.30 * min(1.0, 10.0 / (min_sdc_distance_m + 0.1))     # closer = more interesting
  + 0.25 * min(1.0, num_close_interactions / 50.0)           # more close interactions = more interesting
  + 0.20 * min(1.0, sdc_max_speed_mps / 25.0)               # faster SDC = more interesting
  + 0.15 * min(1.0, num_unique_close_actors / 10.0)          # more actors nearby = more interesting
  + 0.10 * min(1.0, sdc_distance_traveled_m / 200.0)         # more movement = more interesting
)
```

This yields a float in `[0, 1]`. Higher = more interesting scenario.

7. Return a dict with keys: `scenario_id` + all metrics above + `scenario_interest_score`.

**Edge case:** If no SDC track found, or no valid SDC states, return dict with `scenario_id` and all metrics set to `0` or `None`, and `scenario_interest_score = 0.0`.

---

#### `main()`

1. Print header: `"WAYMO VALIDATION LAB — INTERACTION METRICS"`
2. Call `load_silver()`.
3. Get list of unique `scenario_id` values from `scenarios_df`.
4. For each scenario, call `compute_scenario_interactions()`. Print one-line summary per scenario:
   ```
   [1/10] scenario_id=19a486cd29abd7a7  min_dist=2.34m  close_interactions=17  score=0.72
   ```
5. Create `pd.DataFrame` from all result dicts.
6. Sort by `scenario_interest_score` descending.
7. Save to `GOLD_DIR / 'interaction_metrics.parquet'` (create dir if needed).
8. Print summary table showing top 5 scenarios by score.
9. Print `"✅ Interaction metrics saved to data/gold/interaction_metrics.parquet"`.

---

### Step 2 — Output Schema

**File:** `data/gold/interaction_metrics.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `scenario_id` | string | Scenario identifier |
| `min_sdc_distance_m` | float64 | Closest any actor got to SDC |
| `mean_min_sdc_distance_m` | float64 | Average closest distance per timestep |
| `num_close_interactions` | int64 | Count of (timestep, actor) pairs within 5m |
| `num_timesteps_with_close_actor` | int64 | Timesteps with at least one actor < 5m |
| `closest_actor_type` | string | Object type of closest actor |
| `closest_actor_track_id` | string | Track ID of closest actor |
| `sdc_avg_speed_mps` | float64 | SDC mean speed |
| `sdc_max_speed_mps` | float64 | SDC peak speed |
| `sdc_distance_traveled_m` | float64 | Total SDC path length |
| `num_unique_close_actors` | int64 | Distinct actors ever within 5m |
| `scenario_interest_score` | float64 | Composite ranking score [0, 1] |

---

### Step 3 — Testing

Run from project root:

```bash
source .venv/bin/activate
python scripts/waymo_real_parser.py          # ensure silver tables exist
python scripts/compute_interaction_metrics.py
```

**Verify:**
- Script completes without error.
- `data/gold/interaction_metrics.parquet` exists with 10 rows.
- `scenario_interest_score` values are between 0.0 and 1.0.
- `min_sdc_distance_m` values are plausible (typically 1–50 meters).
- `sdc_avg_speed_mps` values are plausible (0–30 m/s).
- Run a quick check:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/gold/interaction_metrics.parquet')
print(df[['scenario_id','min_sdc_distance_m','num_close_interactions','scenario_interest_score']].sort_values('scenario_interest_score', ascending=False).to_string(index=False))
"
```

---

### Step 4 — Update README

Add a subsection under `## 🗃️ Data Schema > ### Gold Tables` in `/Users/rafaelmaranon/waymo-validation-lab/README.md`:

```
**interaction_metrics**
- scenario_id, min_sdc_distance_m, mean_min_sdc_distance_m
- num_close_interactions, num_timesteps_with_close_actor
- closest_actor_type, closest_actor_track_id
- sdc_avg_speed_mps, sdc_max_speed_mps, sdc_distance_traveled_m
- num_unique_close_actors, scenario_interest_score
```

Also add this line under **Quick Start > Run pipeline**:

```
python scripts/compute_interaction_metrics.py
```

(Place it after `compute_basic_metrics.py` and before `validate_outputs.py`.)

---

### What NOT to do

- Do NOT modify any silver tables or the parser.
- Do NOT add new dependencies.
- Do NOT create a web UI.
- Do NOT refactor existing scripts.
- Do NOT change `scenario_metrics.parquet` — the new table is separate.
