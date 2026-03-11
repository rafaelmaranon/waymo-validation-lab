#!/usr/bin/env python3
"""
Compute Basic Metrics Script

Reads silver tables and computes scenario metrics.
Calculates speed, movement, spatial bounds, and other basic metrics.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import math

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import numpy as np
    print("✅ All required imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def calculate_speed(vx: float, vy: float) -> float:
    """Calculate speed from velocity components."""
    if vx is None or vy is None:
        return 0.0
    return math.sqrt(vx**2 + vy**2)

def is_moving_track(track_states: List[Dict], speed_threshold: float = 0.5) -> bool:
    """Determine if a track is moving based on speed threshold."""
    for state in track_states:
        if not state.get('valid', False):
            continue
        speed = calculate_speed(state.get('velocity_x', 0), state.get('velocity_y', 0))
        if speed > speed_threshold:
            return True
    return False

def compute_scenario_metrics(silver_dir: Path) -> pd.DataFrame:
    """Compute metrics for all scenarios."""
    
    print("Loading silver tables...")
    
    # Load silver tables
    scenarios_df = pd.read_parquet(silver_dir / 'scenarios.parquet')
    tracks_df = pd.read_parquet(silver_dir / 'tracks.parquet')
    states_df = pd.read_parquet(silver_dir / 'states.parquet')
    
    print(f"Loaded {len(scenarios_df)} scenarios, {len(tracks_df)} tracks, {len(states_df)} states")
    
    metrics_list = []
    
    # Process each scenario
    for _, scenario in scenarios_df.iterrows():
        scenario_id = scenario['scenario_id']
        
        print(f"Computing metrics for {scenario_id}...")
        
        # Filter data for this scenario
        scenario_tracks = tracks_df[tracks_df['scenario_id'] == scenario_id]
        scenario_states = states_df[states_df['scenario_id'] == scenario_id]
        
        # Basic counts
        num_tracks = len(scenario_tracks)
        num_valid_state_rows = len(scenario_states[scenario_states['valid'] == True])
        
        # Spatial bounds
        valid_states = scenario_states[scenario_states['valid'] == True]
        x_coords = valid_states['x'].dropna()
        y_coords = valid_states['y'].dropna()
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            min_x, max_x = x_coords.min(), x_coords.max()
            min_y, max_y = y_coords.min(), y_coords.max()
        else:
            min_x = max_x = min_y = max_y = None
        
        # Moving tracks calculation
        moving_track_count = 0
        all_speeds = []
        
        for _, track in scenario_tracks.iterrows():
            track_id = track['track_id']
            track_states = scenario_states[scenario_states['track_id'] == track_id].to_dict('records')
            
            # Check if track is moving
            if is_moving_track(track_states):
                moving_track_count += 1
            
            # Collect speeds for average/max calculation
            for state in track_states:
                if state.get('valid', False):
                    speed = calculate_speed(state.get('velocity_x', 0), state.get('velocity_y', 0))
                    all_speeds.append(speed)
        
        # Speed statistics
        if all_speeds:
            avg_speed = np.mean(all_speeds)
            max_speed = np.max(all_speeds)
        else:
            avg_speed = max_speed = 0.0
        
        # Create metrics record
        metrics = {
            'scenario_id': scenario_id,
            'source_file': scenario['source_file'],
            'scenario_index': scenario['scenario_index'],
            'num_tracks': num_tracks,
            'num_valid_state_rows': num_valid_state_rows,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'approx_num_moving_tracks': moving_track_count,
            'avg_speed_mps': avg_speed,
            'max_speed_mps': max_speed
        }
        
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)

def export_metrics(metrics_df: pd.DataFrame, gold_dir: Path):
    """Export metrics to parquet file."""
    
    print("Exporting scenario metrics...")
    
    # Export to parquet
    metrics_df.to_parquet(gold_dir / 'scenario_metrics.parquet', index=False)
    
    print(f"✅ Exported scenario metrics to {gold_dir / 'scenario_metrics.parquet'}")
    print(f"   - {len(metrics_df)} scenarios, {len(metrics_df.columns)} metrics")
    
    # Print summary statistics
    print()
    print("METRICS SUMMARY:")
    print("=" * 40)
    
    numeric_cols = ['num_tracks', 'num_valid_state_rows', 'approx_num_moving_tracks', 
                    'avg_speed_mps', 'max_speed_mps']
    
    for col in numeric_cols:
        if col in metrics_df.columns:
            values = metrics_df[col].dropna()
            if len(values) > 0:
                print(f"{col}:")
                print(f"  Min: {values.min():.2f}")
                print(f"  Max: {values.max():.2f}")
                print(f"  Mean: {values.mean():.2f}")
                print(f"  Median: {values.median():.2f}")
                print()

def main():
    print("=" * 60)
    print("WAYMO VALIDATION LAB - COMPUTE BASIC METRICS")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    silver_dir = project_root / 'data' / 'silver'
    gold_dir = project_root / 'data' / 'gold'
    
    # Ensure output directory exists
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if silver tables exist
    required_files = ['scenarios.parquet', 'tracks.parquet', 'states.parquet']
    missing_files = []
    
    for file_name in required_files:
        file_path = silver_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print("❌ Missing silver tables:")
        for file_name in missing_files:
            print(f"   - {file_name}")
        print()
        print("Please run extract_first_10_scenarios.py first.")
        sys.exit(1)
    
    try:
        # Compute metrics
        metrics_df = compute_scenario_metrics(silver_dir)
        
        # Export metrics
        export_metrics(metrics_df, gold_dir)
        
        print()
        print("=" * 60)
        print("METRICS COMPUTATION COMPLETE")
        print("=" * 60)
        print(f"✅ Computed metrics for {len(metrics_df)} scenarios")
        print(f"✅ Metrics saved to: {gold_dir / 'scenario_metrics.parquet'}")
        
    except Exception as e:
        print(f"❌ Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
