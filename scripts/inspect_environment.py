#!/usr/bin/env python3
"""
Inspect Environment Script

Checks the development environment for Waymo Validation Lab.
Prints Python version, TensorFlow version, Waymo dataset import status,
and verifies TFRecord file existence.
"""

import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("WAYMO VALIDATION LAB - ENVIRONMENT INSPECTION")
    print("=" * 60)
    
    # Python version
    python_version = sys.version
    print(f"Python version: {python_version}")
    
    # Check Python version compatibility
    python_major, python_minor = sys.version_info[:2]
    if python_major == 3 and python_minor >= 10:
        print("✅ Python version is compatible (3.10+)")
    else:
        print("⚠️  Python version may have compatibility issues")
    
    print()
    
    # TensorFlow version
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        print(f"TensorFlow version: {tf_version}")
        
        # Check TensorFlow version
        tf_major = int(tf_version.split('.')[0])
        tf_minor = int(tf_version.split('.')[1])
        if tf_major == 2 and tf_minor >= 12:
            print("✅ TensorFlow version is compatible (2.12+)")
        else:
            print("⚠️  TensorFlow version may have compatibility issues")
    except ImportError as e:
        print("❌ TensorFlow import failed:")
        print(f"   {e}")
    
    print()
    
    # Waymo Open Dataset import
    try:
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset.protos import scenario_pb2
        print("✅ Waymo Open Dataset imports successfully")
        print("   - dataset_pb2 available")
        print("   - scenario_pb2 available")
    except ImportError as e:
        print("❌ Waymo Open Dataset import failed:")
        print(f"   {e}")
        print("   This may affect TFRecord parsing functionality")
    
    print()
    
    # Current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")
    
    # Check if we're in the right directory
    expected_files = ['README.md', 'requirements.txt', 'data/', 'scripts/']
    missing_files = []
    for expected in expected_files:
        expected_path = cwd / expected
        if not expected_path.exists():
            missing_files.append(expected)
    
    if missing_files:
        print(f"⚠️  Missing expected files/directories: {missing_files}")
    else:
        print("✅ Project structure looks correct")
    
    print()
    
    # Check TFRecord file
    tfrecord_dir = cwd / 'data' / 'bronze' / 'waymo_raw'
    tfrecord_files = list(tfrecord_dir.glob('*.tfrecord*'))
    
    print(f"TFRecord directory: {tfrecord_dir}")
    print(f"TFRecord files found: {len(tfrecord_files)}")
    
    if tfrecord_files:
        for tfrecord_file in tfrecord_files:
            size_mb = tfrecord_file.stat().st_size / (1024 * 1024)
            print(f"  - {tfrecord_file.name} ({size_mb:.1f} MB)")
        print("✅ TFRecord files are available for processing")
    else:
        print("❌ No TFRecord files found in data/bronze/waymo_raw/")
        print("   Please copy the source TFRecord file to this directory")
    
    print()
    
    # Check data directories
    data_dirs = [
        'data/bronze/waymo_raw',
        'data/silver', 
        'data/gold',
        'data/exports/scenario_json'
    ]
    
    print("Data directory structure:")
    for data_dir in data_dirs:
        dir_path = cwd / data_dir
        status = "✅" if dir_path.exists() else "❌"
        print(f"  {status} {data_dir}")
    
    print()
    
    # Summary
    print("=" * 60)
    print("ENVIRONMENT SUMMARY")
    print("=" * 60)
    
    issues = []
    
    # Check for major issues
    try:
        import tensorflow as tf
    except ImportError:
        issues.append("TensorFlow not available")
    
    try:
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError:
        issues.append("Waymo Open Dataset not available")
    
    if not tfrecord_files:
        issues.append("No TFRecord files found")
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print()
        print("Some functionality may be limited.")
    else:
        print("✅ Environment looks ready for Waymo data processing!")
        print("You can proceed with the extraction script.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
