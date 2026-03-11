#!/usr/bin/env python3
"""
Debug TFRecord Parser

Analyze the structure of Waymo TFRecord to understand the data format.
"""

import sys
import os
import struct
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def read_tfrecord(tfrecord_path: Path):
    """Read TFRecord file and yield raw records."""
    
    print(f"Reading TFRecord: {tfrecord_path}")
    
    with open(tfrecord_path, 'rb') as f:
        record_count = 0
        
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
            
            record_count += 1
            yield record_count, data

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

def analyze_protobuf_structure(data: bytes, max_depth: int = 3, current_depth: int = 0):
    """Analyze protobuf structure recursively."""
    
    if current_depth >= max_depth:
        return []
    
    fields = []
    offset = 0
    
    while offset < len(data):
        try:
            # Read field key (varint)
            key, offset = parse_protobuf_varint(data, offset)
            
            # Extract field number and wire type
            field_num = key >> 3
            wire_type = key & 0x7
            
            field_info = {
                'field_number': field_num,
                'wire_type': wire_type,
                'wire_type_name': {0: 'varint', 1: '64-bit', 2: 'bytes', 5: '32-bit'}.get(wire_type, 'unknown')
            }
            
            # Handle different wire types
            if wire_type == 2:  # Length-delimited (bytes, string)
                length, offset = parse_protobuf_varint(data, offset)
                field_value = data[offset:offset + length]
                field_info['length'] = length
                
                # Check if this might be a nested message
                if length > 0 and length < 10000:  # Reasonable size for nested message
                    try:
                        nested_fields = analyze_protobuf_structure(field_value, max_depth, current_depth + 1)
                        if nested_fields:
                            field_info['nested_fields'] = nested_fields[:5]  # Limit nested fields
                    except:
                        pass
                
                offset += length
                
            elif wire_type == 0:  # Varint
                value, offset = parse_protobuf_varint(data, offset)
                field_info['value'] = value
                
            elif wire_type == 1:  # 64-bit
                if offset + 8 <= len(data):
                    field_value = data[offset:offset + 8]
                    field_info['value_hex'] = field_value.hex()
                    offset += 8
                else:
                    break
                    
            elif wire_type == 5:  # 32-bit
                if offset + 4 <= len(data):
                    field_value = data[offset:offset + 4]
                    field_info['value_hex'] = field_value.hex()
                    offset += 4
                else:
                    break
            else:
                # Skip unknown wire types
                break
            
            fields.append(field_info)
            
        except Exception as e:
            # Stop parsing on error
            break
    
    return fields

def main():
    print("=" * 60)
    print("WAYMO VALIDATION LAB - TFRECORD STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Setup paths
    dataset_dir = Path.home() / 'datasets' / 'waymo' / 'raw'
    
    # Find TFRecord file
    tfrecord_files = list(dataset_dir.glob('*.tfrecord*'))
    if not tfrecord_files:
        print(f"❌ No TFRecord files found in {dataset_dir}")
        sys.exit(1)
    
    tfrecord_path = tfrecord_files[0]
    print(f"Analyzing: {tfrecord_path}")
    print(f"File size: {tfrecord_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Analyze first few records
    for record_idx, record_data in read_tfrecord(tfrecord_path):
        if record_idx > 3:  # Only analyze first 3 records
            break
        
        print(f"\n--- Record {record_idx} ---")
        print(f"Record size: {len(record_data)} bytes")
        
        # Show first 100 bytes as hex
        hex_preview = record_data[:100].hex()
        print(f"Hex preview: {hex_preview}")
        
        # Try to decode as string
        try:
            string_preview = record_data[:100].decode('utf-8', errors='ignore')
            if string_preview.strip():
                print(f"String preview: {repr(string_preview)}")
        except:
            pass
        
        # Analyze protobuf structure
        try:
            fields = analyze_protobuf_structure(record_data)
            print(f"Protobuf fields found: {len(fields)}")
            
            # Show top-level fields
            for i, field in enumerate(fields[:10]):  # Show first 10 fields
                print(f"  Field {i+1}: #{field['field_number']} ({field['wire_type_name']})", end="")
                if 'length' in field:
                    print(f" - {field['length']} bytes", end="")
                if 'value' in field:
                    print(f" - value: {field['value']}", end="")
                if 'nested_fields' in field:
                    print(f" - nested: {len(field['nested_fields'])} fields", end="")
                print()
                
                # Show nested fields for key fields
                if 'nested_fields' in field and field['field_number'] in [1, 2, 3]:
                    for nested in field['nested_fields'][:3]:
                        print(f"    Nested: #{nested['field_number']} ({nested['wire_type_name']})", end="")
                        if 'length' in nested:
                            print(f" - {nested['length']} bytes", end="")
                        print()
        
        except Exception as e:
            print(f"Error analyzing protobuf: {e}")

if __name__ == "__main__":
    main()
