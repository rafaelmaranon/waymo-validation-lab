# Waymo Parser Assessment

## A. Current Parser Status: `real_waymo_protobuf`

### What is truly decoded from the real Waymo protobuf:
- **scenario_id** — real (e.g. `19a486cd29abd7a7`)
- **sdc_track_index** — real (e.g. `10`)
- **timestamps_seconds** — real (91 steps, 0.0–9.0s)
- **tracks[].id** — real (e.g. `368`, `369`, ...)
- **tracks[].object_type** — real (VEHICLE=947, PEDESTRIAN=50, CYCLIST=2)
- **tracks[].states[].center_x** — real (e.g. `8382.083984`)
- **tracks[].states[].center_y** — real (e.g. `7213.890137`)
- **tracks[].states[].center_z** — real (e.g. `-13.732301`)
- **tracks[].states[].length/width/height** — real (e.g. `4.415 / 1.944 / 1.471`)
- **tracks[].states[].heading** — real (e.g. `-1.557870`)
- **tracks[].states[].velocity_x** — real (e.g. `0.146484`)
- **tracks[].states[].velocity_y** — real (e.g. `-19.819336`)
- **tracks[].states[].valid** — real (`True`/`False`)
- **objects_of_interest** — real
- **map_features** — real (167 features in first scenario, not yet exported)

### What is synthetic:
**Nothing.** All fields in the current pipeline output are decoded from real protobuf.

### What is NOT yet exported (but available in protobuf):
- `map_features` (lane centers, road lines, road edges, stop signs, crosswalks)
- `dynamic_map_states` (traffic signal states)
- `tracks_to_predict` (required prediction targets)
- `current_time_index` (history/future boundary)

## B. What Was the Real Blocker

The previous parsers failed because they were doing **heuristic byte scanning** instead of
using compiled protobuf classes. The actual fix was trivial:

1. Download official `.proto` files from [waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset/tree/master/src/waymo_open_dataset/protos)
2. Compile with `protoc --python_out` (already installed locally at `/usr/local/bin/protoc`)
3. Call `scenario_pb2.Scenario.ParseFromString(raw_bytes)`

**No TensorFlow needed. No Waymo pip package needed. No heavy dependencies.**

The blocker was not "missing schema" or "custom encoding." The `.proto` files were publicly
available on GitHub the entire time. The fix was a 3-step compilation process.

## C. Complexity Assessment

### Moving from previous state to current state:
- **Actual complexity:** LOW
- **Time to implement:** ~30 minutes
- **Steps:** Download 2 proto files → `protoc` compile → `ParseFromString()`
- **Dependencies added:** Zero (protobuf was already installed)

### Remaining complexity for full pipeline:
- **Map feature export:** LOW (data is already decoded, just needs table schema)
- **Dynamic map states:** LOW (same approach)
- **Scaling to all 150 shards:** LOW (same code, just loop over files)

## D. Current Recommendation

**The parsing problem is solved.** All target fields are decoded from real protobuf.

Next priorities should be:
1. Build analytics/ranking on top of real data
2. Add safety/interaction/comfort metrics using real trajectories
3. Scale to more TFRecord shards when needed
4. Export map features if needed for map-based analytics

---

## Evidence

```
scenario_id        : '19a486cd29abd7a7'
sdc_track_index    : 10
num tracks         : 11
num timestamps     : 91
first center_x     : 8382.083984
first center_y     : 7213.890137
first velocity_y   : -19.819336
```

10 scenarios parsed: 999 tracks, 90,909 state rows.
All validations passed. All data is real.
