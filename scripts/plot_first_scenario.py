#!/usr/bin/env python3
"""
Plot First Scenario

Visualizes trajectories from the first scenario in the states.parquet file.
Creates a 2D plot showing actor movements over time.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def plot_scenario_trajectories():
    """
    Load states data and create trajectory plot for first scenario.
    """
    
    # Setup paths
    silver_dir = project_root / 'data' / 'silver'
    exports_dir = project_root / 'data' / 'exports'
    
    # Load states parquet
    states_file = silver_dir / 'states.parquet'
    if not states_file.exists():
        print(f"❌ States file not found: {states_file}")
        return
    
    print(f"Loading states from: {states_file}")
    states_df = pd.read_parquet(states_file)
    print(f"Loaded {len(states_df)} state records")
    
    # Get first scenario
    first_scenario_id = states_df['scenario_id'].iloc[0]
    print(f"Plotting scenario: {first_scenario_id}")
    
    # Filter data for first scenario
    scenario_states = states_df[states_df['scenario_id'] == first_scenario_id].copy()
    print(f"Scenario has {len(scenario_states)} state records")
    
    # Get unique tracks
    tracks = scenario_states['track_id'].unique()
    print(f"Found {len(tracks)} tracks")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Color map for different object types
    colors = {
        'VEHICLE': 'blue',
        'PEDESTRIAN': 'green', 
        'CYCLIST': 'orange',
        'OTHER': 'red'
    }
    
    # Plot each track
    for track_id in tracks:
        track_states = scenario_states[scenario_states['track_id'] == track_id]
        
        if len(track_states) == 0:
            continue
        
        # Sort by timestep to ensure proper trajectory
        track_states = track_states.sort_values('timestep')
        
        # Get track properties
        object_type = track_states['object_type'].iloc[0]
        is_sdc = track_states['is_sdc'].iloc[0]
        
        # Get coordinates
        x_coords = track_states['x'].values
        y_coords = track_states['y'].values
        
        # Skip if all coordinates are the same
        if len(np.unique(x_coords)) == 1 and len(np.unique(y_coords)) == 1:
            print(f"Skipping track {track_id} - no movement detected")
            continue
        
        # Choose color and style
        color = colors.get(object_type, 'black')
        linewidth = 3 if is_sdc else 2
        alpha = 1.0 if is_sdc else 0.7
        
        # Plot trajectory
        label = f"{object_type} ({'SDC' if is_sdc else 'track ' + track_id.split('_')[-1]})"
        plt.plot(x_coords, y_coords, color=color, linewidth=linewidth, alpha=alpha, label=label)
        
        # Mark start and end points
        plt.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=8, alpha=alpha)
        plt.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=8, alpha=alpha)
    
    # Formatting
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(f'Scenario Trajectories: {first_scenario_id}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal')
    
    # Add statistics
    valid_states = scenario_states[scenario_states['valid'] == True]
    if len(valid_states) > 0:
        x_range = valid_states['x'].max() - valid_states['x'].min()
        y_range = valid_states['y'].max() - valid_states['y'].min()
        stats_text = f'X range: {x_range:.1f}m\nY range: {y_range:.1f}m\nValid states: {len(valid_states)}'
        plt.text(0.02, 0.98, stats_text, 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Ensure exports directory exists
    exports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plot_file = exports_dir / 'first_scenario_plot.png'
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plot saved to: {plot_file}")
    
    # Print summary
    object_counts = scenario_states['object_type'].value_counts()
    print(f"📊 Object type distribution: {dict(object_counts)}")
    
    sdc_tracks = scenario_states[scenario_states['is_sdc'] == True]
    print(f"📊 SDC tracks: {len(sdc_tracks['track_id'].unique())}")

def main():
    print("=" * 60)
    print("WAYMO VALIDATION LAB - PLOT FIRST SCENARIO")
    print("=" * 60)
    
    try:
        plot_scenario_trajectories()
        
        print("=" * 60)
        print("PLOTTING COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
