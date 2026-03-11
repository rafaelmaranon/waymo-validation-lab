#!/usr/bin/env python3
"""
Decode One Real Waymo Scenario

Proof-of-work: decode exactly one real Scenario protobuf from TFRecord
using officially compiled Waymo proto classes. Zero heuristics. Zero synthetic data.
"""

import sys
import struct
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'proto'))

# ---------- imports ----------
try:
    from waymo_open_dataset.protos import scenario_pb2
    print("✅ Compiled Waymo protobuf classes imported successfully")
except ImportError as e:
    print(f"❌ Cannot import compiled protobuf classes: {e}")
    sys.exit(1)


def read_first_record(tfrecord_path: Path) -> bytes:
    """Read exactly one TFRecord record (pure Python, no TF)."""
    with open(tfrecord_path, 'rb') as f:
        length_data = f.read(8)
        length = struct.unpack('<Q', length_data)[0]
        f.read(4)  # crc of length
        data = f.read(length)
        f.read(4)  # crc of data
        return data


def main():
    print("=" * 70)
    print("PROOF-OF-WORK: DECODE ONE REAL WAYMO SCENARIO")
    print("=" * 70)

    dataset_dir = Path.home() / 'datasets' / 'waymo' / 'raw'
    tfrecord_files = list(dataset_dir.glob('*.tfrecord*'))
    if not tfrecord_files:
        print(f"❌ No TFRecord files in {dataset_dir}")
        sys.exit(1)

    tfrecord_path = tfrecord_files[0]
    print(f"TFRecord : {tfrecord_path}")
    print(f"File size: {tfrecord_path.stat().st_size / (1024*1024):.1f} MB")

    # ---- read one raw record ----
    print("\nReading first TFRecord record...")
    raw = read_first_record(tfrecord_path)
    print(f"Record size: {len(raw)} bytes")

    # ---- decode with compiled Scenario protobuf ----
    print("\nDecoding with Scenario protobuf class...")
    scenario = scenario_pb2.Scenario()
    try:
        scenario.ParseFromString(raw)
    except Exception as e:
        print(f"❌ Protobuf decode failed: {e}")
        sys.exit(1)

    # ---- print real extracted fields ----
    print("\n" + "=" * 70)
    print("REAL DECODED FIELDS (zero heuristics, zero synthetic)")
    print("=" * 70)

    print(f"scenario_id        : {scenario.scenario_id!r}")
    print(f"sdc_track_index    : {scenario.sdc_track_index}")
    print(f"current_time_index : {scenario.current_time_index}")
    print(f"num timestamps     : {len(scenario.timestamps_seconds)}")
    print(f"num tracks         : {len(scenario.tracks)}")
    print(f"num map_features   : {len(scenario.map_features)}")
    print(f"objects_of_interest: {list(scenario.objects_of_interest)}")

    if scenario.timestamps_seconds:
        print(f"first timestamp    : {scenario.timestamps_seconds[0]:.6f} s")
        print(f"last  timestamp    : {scenario.timestamps_seconds[-1]:.6f} s")

    # ---- print track details ----
    obj_type_map = {0: "UNSET", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}

    print(f"\n--- TRACKS (first 5 of {len(scenario.tracks)}) ---")
    for i, track in enumerate(scenario.tracks[:5]):
        obj_type = obj_type_map.get(track.object_type, f"TYPE_{track.object_type}")
        is_sdc = (i == scenario.sdc_track_index)
        print(f"  Track id={track.id:4d}  type={obj_type:<12s}  "
              f"states={len(track.states):<4d}  {'*** SDC ***' if is_sdc else ''}")

    # ---- print first state of first track ----
    if scenario.tracks and scenario.tracks[0].states:
        first_state = scenario.tracks[0].states[0]
        print(f"\n--- FIRST STATE of Track id={scenario.tracks[0].id} ---")
        print(f"  center_x   : {first_state.center_x:.6f}")
        print(f"  center_y   : {first_state.center_y:.6f}")
        print(f"  center_z   : {first_state.center_z:.6f}")
        print(f"  length     : {first_state.length:.3f}")
        print(f"  width      : {first_state.width:.3f}")
        print(f"  height     : {first_state.height:.3f}")
        print(f"  heading    : {first_state.heading:.6f}")
        print(f"  velocity_x : {first_state.velocity_x:.6f}")
        print(f"  velocity_y : {first_state.velocity_y:.6f}")
        print(f"  valid      : {first_state.valid}")

    # ---- summary ----
    print("\n" + "=" * 70)
    print("RESULT: real_waymo_protobuf ✅")
    print("All fields above are decoded directly from the official Waymo")
    print("Scenario protobuf. Zero heuristics. Zero synthetic generation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
