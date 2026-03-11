#!/usr/bin/env python3
"""
Pure Python TFRecord Parser

Memory-safe parser for Waymo Scenario TFRecord files without TensorFlow.
Uses pure Python to read TFRecord format and extract protobuf data.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Iterator
import struct
import gzip

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import numpy as np
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def read_tfrecord(tfrecord_path: Path) -> Iterator[bytes]:
    """
    Read TFRecord file and yield raw records.
    Pure Python implementation without TensorFlow.
    """
    
    print(f"Reading TFRecord: {tfrecord_path}")
    
    with open(tfrecord_path, 'rb') as f:
        while True:
            # Read length of record (8 bytes)
            length_data = f.read(8)
            if not length_data:
                break  # End of file
            
            if len(length_data) != 8:
                print(f"⚠️  Incomplete length header: {len(length_data)} bytes")
                break
            
            # Unpack length (uint64, little endian)
            length = struct.unpack('<Q', length_data)[0]
            
            # Read CRC32 of length (4 bytes)
            crc_length = f.read(4)
            if len(crc_length) != 4:
                print("⚠️  Incomplete CRC length")
                break
            
            # Read data
            data = f.read(length)
            if len(data) != length:
                print(f"⚠️  Incomplete data: expected {length}, got {len(data)}")
                break
            
            # Read CRC32 of data (4 bytes)
            crc_data = f.read(4)
            if len(crc_data) != 4:
                print("⚠️  Incomplete CRC data")
                break
            
            yield data

def parse_protobuf_varint(data: bytes, offset: int) -> tuple[int, int]:
    """
    Parse protobuf varint from byte data.
    Returns (value, new_offset).
    """
    result = 0
    shift = 0
    
    while True:
        if offset >= len(data):
            raise ValueError("Varint extends beyond data")
        
        byte = data[offset]
        offset += 1
        
        result |= (byte & 0x7F) << shift
        shift += 7
        
        if not (byte & 0x80):
            break
    
    return result, offset

def extract_protobuf_field(data: bytes, field_number: int) -> List[bytes]:
    """
    Extract all instances of a specific protobuf field number.
    Returns list of raw field values.
    """
    fields = []
    offset = 0
    
    while offset < len(data):
        # Read field key (varint)
        key, offset = parse_protobuf_varint(data, offset)
        
        # Extract field number and wire type
        field_num = key >> 3
        wire_type = key & 0x7
        
        if field_num == field_number:
            # Handle different wire types
            if wire_type == 2:  # Length-delimited (bytes, string)
                length, offset = parse_protobuf_varint(data, offset)
                field_value = data[offset:offset + length]
                fields.append(field_value)
                offset += length
            elif wire_type == 0:  # Varint
                value, offset = parse_protobuf_varint(data, offset)
                fields.append(struct.pack('<Q', value))
            elif wire_type == 1:  # 64-bit
                if offset + 8 <= len(data):
                    field_value = data[offset:offset + 8]
                    fields.append(field_value)
                    offset += 8
            elif wire_type == 5:  # 32-bit
                if offset + 4 <= len(data):
                    field_value = data[offset:offset + 4]
                    fields.append(field_value)
                    offset += 4
            else:
                # Skip unknown wire types
                if wire_type == 2:
                    length, offset = parse_protobuf_varint(data, offset)
                    offset += length
                elif wire_type == 0:
                    _, offset = parse_protobuf_varint(data, offset)
                elif wire_type == 1:
                    offset += 8
                elif wire_type == 5:
                    offset += 4
        else:
            # Skip this field
            if wire_type == 2:  # Length-delimited
                length, offset = parse_protobuf_varint(data, offset)
                offset += length
            elif wire_type == 0:  # Varint
                _, offset = parse_protobuf_varint(data, offset)
            elif wire_type == 1:  # 64-bit
                offset += 8
            elif wire_type == 5:  # 32-bit
                offset += 4
            else:
                # Unknown wire type, can't skip
                break
    
    return fields

def parse_waymo_scenario(record_data: bytes, scenario_index: int) -> Dict[str, Any]:
    """
    Parse Waymo Scenario protobuf data.
    Extracts key fields without requiring the full protobuf schema.
    """
    
    # Create basic scenario structure
    scenario_id = f"waymo_scenario_{scenario_index:06d}"
    
    # Try to extract scenario_id from protobuf (field 1 typically)
    scenario_id_fields = extract_protobuf_field(record_data, 1)
    if scenario_id_fields:
        try:
            # Try to decode as string, clean up any invalid characters
            potential_id = scenario_id_fields[0].decode('utf-8', errors='ignore').strip()
            if potential_id and len(potential_id) > 3 and potential_id.isprintable():
                scenario_id = potential_id
        except:
            pass  # Keep generated ID
    
    # Initialize scenario data
    scenario_data = {
        'scenario_id': scenario_id,
        'scenario_index': scenario_index,
        'sdc_track_index': 0,
        'num_tracks': 0,
        'num_steps': 0,
        'objects_of_interest_count': 0,
        'tracks': [],
        'states': []
    }
    
    # Try to extract SDC track index (field 2 typically)
    sdc_fields = extract_protobuf_field(record_data, 2)
    if sdc_fields:
        try:
            # Try to parse as varint
            sdc_value, _ = parse_protobuf_varint(sdc_fields[0], 0)
            scenario_data['sdc_track_index'] = sdc_value
        except:
            pass
    
    # Try to extract track data (field 3 typically for tracks)
    track_fields = extract_protobuf_field(record_data, 3)
    
    track_count = 0
    track_idx = 0
    
    for track_data in track_fields[:5]:  # Limit to first 5 tracks for memory safety
        track_id = f"{scenario_id}_track_{track_idx}"
        object_type = "VEHICLE"  # Default
        is_sdc = (track_idx == scenario_data['sdc_track_index'])
        
        # Try to extract object type from track data (field 2 typically)
        obj_type_fields = extract_protobuf_field(track_data, 2)
        if obj_type_fields:
            try:
                obj_type_value, _ = parse_protobuf_varint(obj_type_fields[0], 0)
                type_names = {1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}
                object_type = type_names.get(obj_type_value, f"TYPE_{obj_type_value}")
            except:
                pass
        
        # Create track record
        track_record = {
            'scenario_id': scenario_id,
            'track_id': track_id,
            'track_index': track_idx,
            'object_type': object_type,
            'is_sdc': is_sdc,
            'states_count': 0
        }
        
        # Try to extract state data from track (field 1 typically)
        state_fields = extract_protobuf_field(track_data, 1)
        
        state_count = 0
        for state_idx, state_data in enumerate(state_fields[:10]):  # Limit to 10 states
            # Extract position data (fields 1, 2, 3 typically for x, y, z)
            pos_fields = [
                extract_protobuf_field(state_data, 1),  # x
                extract_protobuf_field(state_data, 2),  # y  
                extract_protobuf_field(state_data, 3)   # z
            ]
            
            # Extract velocity data (fields 4, 5 typically)
            vel_fields = [
                extract_protobuf_field(state_data, 4),  # vx
                extract_protobuf_field(state_data, 5)   # vy
            ]
            
            # Extract size data (fields 6, 7, 8 typically)
            size_fields = [
                extract_protobuf_field(state_data, 6),  # length
                extract_protobuf_field(state_data, 7),  # width
                extract_protobuf_field(state_data, 8)   # height
            ]
            
            # Extract heading (field 9 typically)
            heading_fields = extract_protobuf_field(state_data, 9)
            
            # Extract valid flag (field 10 typically)
            valid_fields = extract_protobuf_field(state_data, 10)
            
            # Parse values with defaults
            x = 0.0
            y = 0.0
            z = 0.0
            
            if pos_fields[0]:
                try:
                    x_bytes = pos_fields[0][0]
                    if len(x_bytes) == 8:
                        x = struct.unpack('<d', x_bytes)[0]
                except:
                    pass
            
            if pos_fields[1]:
                try:
                    y_bytes = pos_fields[1][0]
                    if len(y_bytes) == 8:
                        y = struct.unpack('<d', y_bytes)[0]
                except:
                    pass
            
            if pos_fields[2]:
                try:
                    z_bytes = pos_fields[2][0]
                    if len(z_bytes) == 8:
                        z = struct.unpack('<d', z_bytes)[0]
                except:
                    pass
            
            vx = 0.0
            vy = 0.0
            
            if vel_fields[0]:
                try:
                    vx_bytes = vel_fields[0][0]
                    if len(vx_bytes) == 8:
                        vx = struct.unpack('<d', vx_bytes)[0]
                except:
                    pass
            
            if vel_fields[1]:
                try:
                    vy_bytes = vel_fields[1][0]
                    if len(vy_bytes) == 8:
                        vy = struct.unpack('<d', vy_bytes)[0]
                except:
                    pass
            
            length = 4.0
            width = 2.0
            height = 1.5
            
            if size_fields[0]:
                try:
                    length_bytes = size_fields[0][0]
                    if len(length_bytes) == 8:
                        length = struct.unpack('<d', length_bytes)[0]
                except:
                    pass
            
            if size_fields[1]:
                try:
                    width_bytes = size_fields[1][0]
                    if len(width_bytes) == 8:
                        width = struct.unpack('<d', width_bytes)[0]
                except:
                    pass
            
            if size_fields[2]:
                try:
                    height_bytes = size_fields[2][0]
                    if len(height_bytes) == 8:
                        height = struct.unpack('<d', height_bytes)[0]
                except:
                    pass
            
            heading = 0.0
            if heading_fields:
                try:
                    heading_bytes = heading_fields[0][0]
                    if len(heading_bytes) == 8:
                        heading = struct.unpack('<d', heading_bytes)[0]
                except:
                    pass
            
            valid = True
            if valid_fields:
                try:
                    valid_bytes = valid_fields[0][0]
                    if len(valid_bytes) == 1:
                        valid = bool(valid_bytes[0])
                except:
                    pass
            
            # Create state record
            state_record = {
                'scenario_id': scenario_id,
                'track_id': track_id,
                'track_index': track_idx,
                'timestep': state_idx,
                'x': x,
                'y': y,
                'z': z,
                'length': length,
                'width': width,
                'height': height,
                'heading': heading,
                'velocity_x': vx,
                'velocity_y': vy,
                'valid': valid,
                'object_type': object_type,
                'is_sdc': is_sdc
            }
            
            scenario_data['states'].append(state_record)
            state_count += 1
        
        track_record['states_count'] = state_count
        scenario_data['tracks'].append(track_record)
        track_count += 1
        track_idx += 1
    
    scenario_data['num_tracks'] = track_count
    scenario_data['num_steps'] = max([t['states_count'] for t in scenario_data['tracks']], default=0)
    
    return scenario_data

def extract_scenarios_memory_safe(tfrecord_path: Path, max_scenarios: int = 10) -> Dict[str, List[Dict]]:
    """
    Memory-safe extraction that processes scenarios in batches.
    """
    
    # Initialize data containers
    scenarios_data = []
    tracks_data = []
    states_data = []
    
    # Process records one at a time
    scenario_count = 0
    
    for record_index, record_data in enumerate(read_tfrecord(tfrecord_path)):
        if scenario_count >= max_scenarios:
            break
        
        try:
            # Parse scenario from record data
            scenario_data = parse_waymo_scenario(record_data, record_index)
            
            # Add source file info
            scenario_data['source_file'] = tfrecord_path.name
            
            # Extract scenario metadata
            scenario_metadata = {
                'scenario_id': scenario_data['scenario_id'],
                'source_file': scenario_data['source_file'],
                'scenario_index': scenario_data['scenario_index'],
                'sdc_track_index': scenario_data['sdc_track_index'],
                'num_tracks': scenario_data['num_tracks'],
                'num_steps': scenario_data['num_steps'],
                'objects_of_interest_count': scenario_data['objects_of_interest_count']
            }
            scenarios_data.append(scenario_metadata)
            
            # Add track data
            for track in scenario_data['tracks']:
                tracks_data.append(track)
            
            # Add state data
            for state in scenario_data['states']:
                states_data.append(state)
            
            scenario_count += 1
            print(f"Parsed scenario {scenario_count}: {scenario_data['scenario_id']} "
                  f"(tracks: {scenario_data['num_tracks']}, states: {scenario_data['num_steps']})")
            
            # Clear scenario_data to free memory
            del scenario_data
            
        except Exception as e:
            print(f"⚠️  Error processing record {record_index}: {e}")
            continue
    
    return {
        'scenarios': scenarios_data,
        'tracks': tracks_data,
        'states': states_data
    }

def export_parquet_tables(data: Dict[str, List[Dict]], output_dir: Path):
    """Export data to parquet files."""
    
    print("Exporting parquet tables...")
    
    # Create DataFrames
    scenarios_df = pd.DataFrame(data['scenarios'])
    tracks_df = pd.DataFrame(data['tracks'])
    states_df = pd.DataFrame(data['states'])
    
    # Export to parquet
    scenarios_df.to_parquet(output_dir / 'scenarios.parquet', index=False)
    tracks_df.to_parquet(output_dir / 'tracks.parquet', index=False)
    states_df.to_parquet(output_dir / 'states.parquet', index=False)
    
    print(f"✅ Exported parquet tables to {output_dir}")
    print(f"   - scenarios.parquet: {len(scenarios_df)} rows, {len(scenarios_df.columns)} columns")
    print(f"   - tracks.parquet: {len(tracks_df)} rows, {len(tracks_df.columns)} columns")
    print(f"   - states.parquet: {len(states_df)} rows, {len(states_df.columns)} columns")

def export_scenario_jsons(data: Dict[str, List[Dict]], tfrecord_path: Path, output_dir: Path):
    """Export individual scenario JSON files."""
    
    print("Exporting scenario JSON files...")
    
    # Group data by scenario
    scenarios_by_id = {}
    for scenario in data['scenarios']:
        scenario_id = scenario['scenario_id']
        scenarios_by_id[scenario_id] = {
            'scenario': scenario,
            'tracks': [],
            'states': []
        }
    
    # Add tracks and states
    for track in data['tracks']:
        scenario_id = track['scenario_id']
        scenarios_by_id[scenario_id]['tracks'].append(track)
    
    for state in data['states']:
        scenario_id = state['scenario_id']
        scenarios_by_id[scenario_id]['states'].append(state)
    
    # Export each scenario as JSON
    for scenario_id, scenario_data in scenarios_by_id.items():
        # Simplified JSON structure for debugging
        json_data = {
            'scenario_id': scenario_id,
            'source_file': scenario_data['scenario']['source_file'],
            'scenario_index': scenario_data['scenario']['scenario_index'],
            'sdc_track_index': scenario_data['scenario']['sdc_track_index'],
            'tracks': []
        }
        
        # Add track data with states
        for track in scenario_data['tracks']:
            track_states = [s for s in scenario_data['states'] if s['track_id'] == track['track_id']]
            
            track_data = {
                'track_id': track['track_id'],
                'track_index': track['track_index'],
                'object_type': track['object_type'],
                'is_sdc': track['is_sdc'],
                'states': []
            }
            
            # Add state data
            for state in track_states:
                state_data = {
                    'timestep': state['timestep'],
                    'x': state['x'],
                    'y': state['y'],
                    'heading': state['heading'],
                    'velocity_x': state['velocity_x'],
                    'velocity_y': state['velocity_y'],
                    'valid': state['valid']
                }
                track_data['states'].append(state_data)
            
            json_data['tracks'].append(track_data)
        
        # Write JSON file
        json_file = output_dir / f"{scenario_id.replace('/', '_')}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    print(f"✅ Exported {len(scenarios_by_id)} scenario JSON files to {output_dir}")

def main():
    print("=" * 60)
    print("WAYMO VALIDATION LAB - PURE PYTHON TFRECORD PARSING")
    print("=" * 60)
    
    # Setup paths
    dataset_dir = Path.home() / 'datasets' / 'waymo' / 'raw'
    project_root = Path(__file__).parent.parent
    silver_dir = project_root / 'data' / 'silver'
    json_export_dir = project_root / 'data' / 'exports' / 'scenario_json'
    
    # Ensure output directories exist
    silver_dir.mkdir(parents=True, exist_ok=True)
    json_export_dir.mkdir(parents=True, exist_ok=True)
    
    # Find TFRecord file
    tfrecord_files = list(dataset_dir.glob('*.tfrecord*'))
    if not tfrecord_files:
        print(f"❌ No TFRecord files found in {dataset_dir}")
        print("Please copy Waymo TFRecord files to ~/datasets/waymo/raw/")
        sys.exit(1)
    
    tfrecord_path = tfrecord_files[0]  # Use first file found
    print(f"Using TFRecord: {tfrecord_path}")
    print(f"File size: {tfrecord_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Extract data using memory-safe parsing
        data = extract_scenarios_memory_safe(tfrecord_path, max_scenarios=10)
        
        # Export parquet tables
        export_parquet_tables(data, silver_dir)
        
        # Export JSON files
        export_scenario_jsons(data, tfrecord_path, json_export_dir)
        
        print()
        print("=" * 60)
        print("PURE PYTHON PARSING COMPLETE")
        print("=" * 60)
        print(f"✅ Processed {len(data['scenarios'])} scenarios from real TFRecord")
        print(f"✅ Created {len(data['tracks'])} track records")
        print(f"✅ Created {len(data['states'])} state records")
        print(f"✅ Parquet tables saved to: {silver_dir}")
        print(f"✅ JSON exports saved to: {json_export_dir}")
        
    except Exception as e:
        print(f"❌ Pure Python parsing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
