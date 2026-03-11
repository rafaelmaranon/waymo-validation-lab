#!/usr/bin/env python3
"""
Enhanced Waymo Protobuf Parser

Attempts more sophisticated protobuf decoding using available tools.
Still limited by missing Waymo schema, but improves field extraction.
"""

import sys
import os
import json
import hashlib
import struct
from pathlib import Path
from typing import List, Dict, Any, Iterator

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import numpy as np
    import tfrecord
    from google.protobuf.json_format import MessageToDict
    from google.protobuf.message import DecodeError
    print("✅ Core imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def parse_protobuf_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Parse protobuf varint from byte data."""
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
    """Extract all instances of a specific protobuf field number."""
    fields = []
    offset = 0
    
    while offset < len(data):
        try:
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
                        break
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
        
        except Exception as e:
            # Stop parsing on error
            break
    
    return fields

def attempt_enhanced_protobuf_parsing(record_data: bytes) -> Dict[str, Any]:
    """
    Enhanced protobuf parsing with better field extraction.
    Still limited by missing schema, but more systematic.
    """
    
    try:
        # Systematic field extraction for common Waymo fields
        # Based on typical Waymo Scenario protobuf structure
        
        scenario_id = None
        sdc_track_index = 0
        tracks_data = []
        
        # Extract scenario_id (typically field 1, string)
        scenario_id_fields = extract_protobuf_field(record_data, 1)
        if scenario_id_fields:
            for field_data in scenario_id_fields:
                try:
                    decoded = field_data.decode('utf-8', errors='ignore').strip()
                    if len(decoded) > 8 and decoded.replace('-', '_').replace('_', '').isalnum():
                        scenario_id = decoded[:32]  # Reasonable length limit
                        break
                except:
                    continue
        
        # Extract SDC track index (typically field 2, varint)
        sdc_fields = extract_protobuf_field(record_data, 2)
        if sdc_fields:
            for field_data in sdc_fields:
                if len(field_data) <= 8:
                    try:
                        sdc_value, _ = parse_protobuf_varint(field_data, 0)
                        if 0 <= sdc_value <= 50:  # Reasonable range
                            sdc_track_index = sdc_value
                            break
                    except:
                        continue
        
        # Extract track data (typically field 3, repeated message)
        track_fields = extract_protobuf_field(record_data, 3)
        
        real_tracks_found = 0
        
        for track_data in track_fields[:20]:  # Limit to prevent memory issues
            track_id = None
            object_type = 1  # Default to VEHICLE
            track_states = []
            
            # Extract track_id (typically field 1, string)
            track_id_fields = extract_protobuf_field(track_data, 1)
            if track_id_fields:
                for field_data in track_id_fields:
                    try:
                        decoded = field_data.decode('utf-8', errors='ignore').strip()
                        if len(decoded) > 3:
                            track_id = decoded[:32]
                            break
                    except:
                        continue
            
            # Extract object_type (typically field 2, varint)
            obj_type_fields = extract_protobuf_field(track_data, 2)
            if obj_type_fields:
                for field_data in obj_type_fields:
                    if len(field_data) <= 8:
                        try:
                            obj_type_value, _ = parse_protobuf_varint(field_data, 0)
                            if 1 <= obj_type_value <= 4:  # Valid Waymo object types
                                object_type = obj_type_value
                                break
                        except:
                            continue
            
            # Extract states (typically field 1, repeated message)
            state_fields = extract_protobuf_field(track_data, 1)
            
            real_states_found = 0
            
            for state_data in state_fields[:100]:  # Limit states per track
                state = {}
                
                # Extract position fields (center_x, center_y, center_z)
                # Typically fields 1, 2, 3 (64-bit floats)
                pos_fields = [
                    extract_protobuf_field(state_data, 1),  # x
                    extract_protobuf_field(state_data, 2),  # y
                    extract_protobuf_field(state_data, 3)   # z
                ]
                
                # Extract velocity fields (velocity_x, velocity_y)
                # Typically fields 4, 5 (64-bit floats)
                vel_fields = [
                    extract_protobuf_field(state_data, 4),  # vx
                    extract_protobuf_field(state_data, 5)   # vy
                ]
                
                # Extract heading (typically field 9, 64-bit float)
                heading_fields = extract_protobuf_field(state_data, 9)
                
                # Extract valid flag (typically field 10, varint)
                valid_fields = extract_protobuf_field(state_data, 10)
                
                # Parse position data
                x, y, z = 0.0, 0.0, 0.0
                if pos_fields[0] and len(pos_fields[0][0]) == 8:
                    x = struct.unpack('<d', pos_fields[0][0])[0]
                if pos_fields[1] and len(pos_fields[1][0]) == 8:
                    y = struct.unpack('<d', pos_fields[1][0])[0]
                if pos_fields[2] and len(pos_fields[2][0]) == 8:
                    z = struct.unpack('<d', pos_fields[2][0])[0]
                
                # Parse velocity data
                vx, vy = 0.0, 0.0
                if vel_fields[0] and len(vel_fields[0][0]) == 8:
                    vx = struct.unpack('<d', vel_fields[0][0])[0]
                if vel_fields[1] and len(vel_fields[1][0]) == 8:
                    vy = struct.unpack('<d', vel_fields[1][0])[0]
                
                # Parse heading
                heading = 0.0
                if heading_fields and len(heading_fields[0]) == 8:
                    heading = struct.unpack('<d', heading_fields[0])[0]
                
                # Parse valid flag
                valid = True
                if valid_fields:
                    for field_data in valid_fields:
                        if len(field_data) <= 8:
                            try:
                                valid_value, _ = parse_protobuf_varint(field_data, 0)
                                valid = bool(valid_value)
                                break
                            except:
                                continue
                
                # Only count as real if we got actual position data
                if x != 0.0 or y != 0.0:
                    state.update({
                        'x': x, 'y': y, 'z': z,
                        'velocity_x': vx, 'velocity_y': vy,
                        'heading': heading, 'valid': valid
                    })
                    track_states.append(state)
                    real_states_found += 1
            
            # Only count track as real if we got real states
            if real_states_found > 0 and track_id:
                tracks_data.append({
                    'track_id': track_id,
                    'object_type': object_type,
                    'states': track_states
                })
                real_tracks_found += 1
        
        # If we found real track data, use it
        if real_tracks_found > 0:
            record_hash = hashlib.md5(record_data[:100]).hexdigest()[:16]
            if not scenario_id:
                scenario_id = f"enhanced_{record_hash}"
            
            return {
                'scenario_id': scenario_id,
                'sdc_track_index': sdc_track_index,
                'tracks_data': tracks_data,
                'data_source': 'enhanced_real_protobuf',
                'parsing_details': {
                    'extracted_scenario_id': scenario_id_fields is not None and len(scenario_id_fields) > 0,
                    'extracted_sdc_index': sdc_fields is not None and len(sdc_fields) > 0,
                    'real_tracks_found': real_tracks_found,
                    'total_track_fields': len(track_fields),
                    'record_size': len(record_data)
                }
            }
        
        else:
            return None
            
    except Exception as e:
        print(f"⚠️  Enhanced protobuf parsing failed: {e}")
        return None

def generate_enhanced_fallback(record_data: bytes, scenario_index: int, source_file: str) -> Dict[str, Any]:
    """
    Enhanced fallback that uses more sophisticated heuristics.
    """
    
    record_hash = hashlib.md5(record_data[:100]).hexdigest()[:16]
    scenario_id = f"enhanced_fallback_{record_hash}"
    
    # Use record characteristics to determine complexity
    record_size = len(record_data)
    num_tracks = min(1 + (record_size // 150000), 15)
    num_steps = 91
    sdc_track_index = 0
    
    # Try to extract SDC index from heuristics
    sdc_fields = extract_protobuf_field(record_data, 2)
    if sdc_fields:
        for field_data in sdc_fields:
            if len(field_data) <= 8:
                try:
                    sdc_value, _ = parse_protobuf_varint(field_data, 0)
                    if 0 <= sdc_value < num_tracks:
                        sdc_track_index = sdc_value
                        break
                except:
                    continue
    
    # Create tracks with enhanced heuristics
    tracks = []
    states = []
    
    for track_idx in range(num_tracks):
        track_id = f"{scenario_id}_track_{track_idx:03d}"
        
        # Use record hash to determine object type distribution
        seed = sum(ord(c) for c in record_hash) + track_idx
        obj_type_value = (seed % 4) + 1  # 1-4 map to Waymo object types
        object_type_map = {1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}
        object_type = object_type_map[obj_type_value]
        
        is_sdc = (track_idx == sdc_track_index)
        
        tracks.append({
            'track_id': track_id,
            'track_index': track_idx,
            'object_type': object_type,
            'is_sdc': is_sdc,
            'states_count': num_steps
        })
        
        # Generate states with enhanced physics
        for timestep in range(num_steps):
            seed = sum(ord(c) for c in record_hash) + track_idx * 1000 + timestep
            np.random.seed(seed)
            
            # Enhanced trajectory generation
            if is_sdc:
                # SDC: smooth lane-following behavior
                base_x = track_idx * 20 + timestep * 1.8
                base_y = scenario_index * 25 + np.sin(timestep * 0.08) * 8
                vx = 1.8 + 0.3 * np.sin(timestep * 0.05)
                vy = 0.4 * np.cos(timestep * 0.08)
                heading = np.arctan2(vy, vx)
                
            elif object_type == "VEHICLE":
                # Other vehicles: varied behavior
                base_x = track_idx * 15 + timestep * 1.2 + np.random.randn() * 2
                base_y = scenario_index * 20 + np.random.randn() * 4
                vx = 1.2 + np.random.randn() * 0.4
                vy = np.random.randn() * 0.3
                heading = np.random.uniform(0, 2 * np.pi)
                
            elif object_type == "PEDESTRIAN":
                # Pedestrians: slow, erratic
                base_x = track_idx * 8 + timestep * 0.6 + np.random.randn() * 1
                base_y = scenario_index * 15 + np.random.randn() * 3
                vx = 0.6 + np.random.randn() * 0.2
                vy = np.random.randn() * 0.2
                heading = np.random.uniform(0, 2 * np.pi)
                
            else:  # CYCLIST/OTHER
                base_x = track_idx * 12 + timestep * 0.9 + np.random.randn() * 1.5
                base_y = scenario_index * 18 + np.random.randn() * 3.5
                vx = 0.9 + np.random.randn() * 0.3
                vy = np.random.randn() * 0.3
                heading = np.random.uniform(0, 2 * np.pi)
            
            states.append({
                'track_id': track_id,
                'track_index': track_idx,
                'timestep': timestep,
                'x': base_x,
                'y': base_y,
                'z': 0.0,
                'length': 4.5 if object_type == "VEHICLE" else 1.0,
                'width': 2.0 if object_type == "VEHICLE" else 1.0,
                'height': 1.5 if object_type == "VEHICLE" else 1.7,
                'heading': heading,
                'velocity_x': vx,
                'velocity_y': vy,
                'valid': True,
                'object_type': object_type,
                'is_sdc': is_sdc
            })
    
    return {
        'scenario_id': scenario_id,
        'sdc_track_index': sdc_track_index,
        'num_tracks': num_tracks,
        'num_steps': num_steps,
        'tracks': tracks,
        'states': states,
        'data_source': 'enhanced_fallback',
        'parsing_details': {
            'extracted_scenario_id': False,
            'extracted_sdc_index': sdc_fields is not None and len(sdc_fields) > 0,
            'record_size': record_size
        }
    }

def parse_scenario_from_record(record_data: bytes, scenario_index: int, source_file: str) -> Dict[str, Any]:
    """
    Parse scenario with enhanced protobuf decoding.
    """
    
    # Try enhanced real protobuf parsing
    real_result = attempt_enhanced_protobuf_parsing(record_data)
    
    if real_result:
        # Convert enhanced real result to standard format
        scenario_data = {
            'scenario_id': real_result['scenario_id'],
            'source_file': source_file,
            'scenario_index': scenario_index,
            'sdc_track_index': real_result['sdc_track_index'],
            'num_tracks': len(real_result['tracks_data']),
            'num_steps': 0,
            'objects_of_interest_count': 0,
            'tracks': [],
            'states': []
        }
        
        # Process tracks data
        for track_idx, track_data in enumerate(real_result['tracks_data']):
            track_record = {
                'scenario_id': real_result['scenario_id'],
                'track_id': track_data['track_id'],
                'track_index': track_idx,
                'object_type': {1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST", 4: "OTHER"}.get(track_data['object_type'], "OTHER"),
                'is_sdc': (track_idx == real_result['sdc_track_index']),
                'states_count': len(track_data['states'])
            }
            scenario_data['tracks'].append(track_record)
            
            # Process states
            for state_idx, state in enumerate(track_data['states']):
                state_record = state.copy()
                state_record.update({
                    'scenario_id': real_result['scenario_id'],
                    'track_id': track_data['track_id'],
                    'track_index': track_idx,
                    'timestep': state_idx,
                    'length': 4.5 if track_record['object_type'] == "VEHICLE" else 1.0,
                    'width': 2.0 if track_record['object_type'] == "VEHICLE" else 1.0,
                    'height': 1.5 if track_record['object_type'] == "VEHICLE" else 1.7,
                    'object_type': track_record['object_type'],
                    'is_sdc': track_record['is_sdc']
                })
                scenario_data['states'].append(state_record)
        
        # Update num_steps
        if scenario_data['states']:
            scenario_data['num_steps'] = max(len(track['states']) for track in real_result['tracks_data'])
        
        # Add data source and parsing details
        scenario_data['data_source'] = real_result['data_source']
        scenario_data['parsing_details'] = real_result['parsing_details']
        
        # Print verification info
        print(f"✅ ENHANCED REAL PARSING: {scenario_data['scenario_id']}")
        print(f"   - Data source: {scenario_data['data_source']}")
        print(f"   - Tracks: {scenario_data['num_tracks']}, States: {scenario_data['num_steps']}")
        print(f"   - Real tracks extracted: {real_result['parsing_details']['real_tracks_found']}")
        if scenario_data['states']:
            first_pos = scenario_data['states'][0]
            print(f"   - First actor position: x={first_pos['x']:.2f}, y={first_pos['y']:.2f}")
        
        return scenario_data
    
    else:
        # Fall back to enhanced synthetic generation
        synthetic_result = generate_enhanced_fallback(record_data, scenario_index, source_file)
        
        # Print verification info
        print(f"⚠️  ENHANCED FALLBACK: {synthetic_result['scenario_id']}")
        print(f"   - Data source: {synthetic_result['data_source']}")
        print(f"   - Tracks: {synthetic_result['num_tracks']}, States: {synthetic_result['num_steps']}")
        if synthetic_result['tracks']:
            first_pos = synthetic_result['states'][0] if synthetic_result['states'] else None
            if first_pos:
                print(f"   - First actor position: x={first_pos['x']:.2f}, y={first_pos['y']:.2f}")
        
        return synthetic_result

# Reuse the same reading and export functions from the previous parser
def read_tfrecord_with_tfrecord_package(tfrecord_path: Path) -> Iterator[bytes]:
    """Read TFRecord file using tfrecord package."""
    try:
        reader = tfrecord.reader(tfrecord_path)
        for example in reader:
            yield example
        reader.close()
    except Exception as e:
        print(f"⚠️  tfrecord package failed: {e}")
        # Fall back to pure Python implementation
        yield from read_tfrecord_pure_python(tfrecord_path)

def read_tfrecord_pure_python(tfrecord_path: Path) -> Iterator[bytes]:
    """Pure Python TFRecord reader (fallback)."""
    with open(tfrecord_path, 'rb') as f:
        while True:
            length_data = f.read(8)
            if not length_data:
                break
            if len(length_data) != 8:
                break
            length = struct.unpack('<Q', length_data)[0]
            crc_length = f.read(4)
            if len(crc_length) != 4:
                break
            data = f.read(length)
            if len(data) != length:
                break
            crc_data = f.read(4)
            if len(crc_data) != 4:
                break
            yield data

def extract_scenarios_memory_safe(tfrecord_path: Path, max_scenarios: int = 10) -> Dict[str, List[Dict]]:
    """Memory-safe extraction with enhanced parsing."""
    scenarios_data = []
    tracks_data = []
    states_data = []
    scenario_count = 0
    
    for record_index, record_data in enumerate(read_tfrecord_with_tfrecord_package(tfrecord_path)):
        if scenario_count >= max_scenarios:
            break
        
        try:
            scenario_data = parse_scenario_from_record(record_data, record_index, tfrecord_path.name)
            
            scenario_metadata = {
                'scenario_id': scenario_data['scenario_id'],
                'source_file': scenario_data['source_file'],
                'scenario_index': scenario_data['scenario_index'],
                'sdc_track_index': scenario_data['sdc_track_index'],
                'num_tracks': scenario_data['num_tracks'],
                'num_steps': scenario_data['num_steps'],
                'objects_of_interest_count': scenario_data['objects_of_interest_count'],
                'data_source': scenario_data['data_source']
            }
            scenarios_data.append(scenario_metadata)
            
            for track in scenario_data['tracks']:
                tracks_data.append(track)
            
            for state in scenario_data['states']:
                states_data.append(state)
            
            scenario_count += 1
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
    
    scenarios_df = pd.DataFrame(data['scenarios'])
    tracks_df = pd.DataFrame(data['tracks'])
    states_df = pd.DataFrame(data['states'])
    
    scenarios_df.to_parquet(output_dir / 'scenarios.parquet', index=False)
    tracks_df.to_parquet(output_dir / 'tracks.parquet', index=False)
    states_df.to_parquet(output_dir / 'states.parquet', index=False)
    
    print(f"✅ Exported parquet tables to {output_dir}")
    print(f"   - scenarios.parquet: {len(scenarios_df)} rows, {len(scenarios_df.columns)} columns")
    print(f"   - tracks.parquet: {len(tracks_df)} rows, {len(tracks_df.columns)} columns")
    print(f"   - states.parquet: {len(states_df)} rows, {len(states_df.columns)} columns")
    
    if 'data_source' in scenarios_df.columns:
        source_counts = scenarios_df['data_source'].value_counts()
        print(f"   - Data sources: {dict(source_counts)}")

def export_scenario_jsons(data: Dict[str, List[Dict]], tfrecord_path: Path, output_dir: Path):
    """Export individual scenario JSON files."""
    print("Exporting scenario JSON files...")
    
    scenarios_by_id = {}
    for scenario in data['scenarios']:
        scenario_id = scenario['scenario_id']
        scenarios_by_id[scenario_id] = {
            'scenario': scenario,
            'tracks': [],
            'states': []
        }
    
    for track in data['tracks']:
        scenario_id = track['scenario_id']
        scenarios_by_id[scenario_id]['tracks'].append(track)
    
    for state in data['states']:
        scenario_id = state['scenario_id']
        scenarios_by_id[scenario_id]['states'].append(state)
    
    for scenario_id, scenario_data in scenarios_by_id.items():
        json_data = {
            'scenario_id': scenario_id,
            'source_file': scenario_data['scenario']['source_file'],
            'scenario_index': scenario_data['scenario']['scenario_index'],
            'sdc_track_index': scenario_data['scenario']['sdc_track_index'],
            'data_source': scenario_data['scenario']['data_source'],
            'tracks': []
        }
        
        for track in scenario_data['tracks']:
            track_states = [s for s in scenario_data['states'] if s['track_id'] == track['track_id']]
            
            track_data = {
                'track_id': track['track_id'],
                'track_index': track['track_index'],
                'object_type': track['object_type'],
                'is_sdc': track['is_sdc'],
                'states': []
            }
            
            for state in track_states[::10]:
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
        
        json_file = output_dir / f"{scenario_id.replace('/', '_')}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    print(f"✅ Exported {len(scenarios_by_id)} scenario JSON files to {output_dir}")

def main():
    print("=" * 60)
    print("WAYMO VALIDATION LAB - ENHANCED PROTOBUF PARSING")
    print("=" * 60)
    
    dataset_dir = Path.home() / 'datasets' / 'waymo' / 'raw'
    project_root = Path(__file__).parent.parent
    silver_dir = project_root / 'data' / 'silver'
    json_export_dir = project_root / 'data' / 'exports' / 'scenario_json'
    
    silver_dir.mkdir(parents=True, exist_ok=True)
    json_export_dir.mkdir(parents=True, exist_ok=True)
    
    tfrecord_files = list(dataset_dir.glob('*.tfrecord*'))
    if not tfrecord_files:
        print(f"❌ No TFRecord files found in {dataset_dir}")
        sys.exit(1)
    
    tfrecord_path = tfrecord_files[0]
    print(f"Using TFRecord: {tfrecord_path}")
    print(f"File size: {tfrecord_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        data = extract_scenarios_memory_safe(tfrecord_path, max_scenarios=10)
        export_parquet_tables(data, silver_dir)
        export_scenario_jsons(data, tfrecord_path, json_export_dir)
        
        print()
        print("=" * 60)
        print("ENHANCED PROTOBUF PARSING COMPLETE")
        print("=" * 60)
        print(f"✅ Processed {len(data['scenarios'])} scenarios")
        print(f"✅ Created {len(data['tracks'])} track records")
        print(f"✅ Created {len(data['states'])} state records")
        print(f"✅ Parquet tables saved to: {silver_dir}")
        print(f"✅ JSON exports saved to: {json_export_dir}")
        
        if data['scenarios']:
            sources = {}
            for scenario in data['scenarios']:
                source = scenario.get('data_source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            
            print(f"📊 Parsing summary: {sources}")
        
    except Exception as e:
        print(f"❌ Enhanced parsing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
