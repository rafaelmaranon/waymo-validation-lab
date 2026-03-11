Version: v1.0_real_waymo_protobuf_parser

Status:
Working parser decoding real Waymo Scenario protobuf
using compiled proto files.

Capabilities:
- TFRecord ingestion
- Scenario.ParseFromString()
- normalized tables (scenarios, tracks, states)
- parquet exports
- visual validation

Dataset example:
scenario_id: 19a486cd29abd7a7
tracks: 11
timesteps: 91

Purpose of snapshot:
Lock stable ingestion pipeline before adding analytics layer.
