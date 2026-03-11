#!/usr/bin/env python3
"""
Animate First Scenario

Creates an animated GIF showing actor movements over time.
Visualizes the dynamic nature of the scenario.
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
    import matplotlib.animation as animation
    import numpy as np
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def animate_scenario():
    """
    Load states data and create animated GIF for first scenario.
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
    print(f"Animating scenario: {first_scenario_id}")
    
    # Filter data for first scenario
    scenario_states = states_df[states_df['scenario_id'] == first_scenario_id].copy()
    print(f"Scenario has {len(scenario_states)} state records")
    
    # Get unique tracks and timesteps
    tracks = scenario_states['track_id'].unique()
    timesteps = sorted(scenario_states['timestep'].unique())
    print(f"Found {len(tracks)} tracks, {len(timesteps)} timesteps")
    
    # Color map for different object types
    colors = {
        'VEHICLE': 'blue',
        'PEDESTRIAN': 'green', 
        'CYCLIST': 'orange',
        'OTHER': 'red'
    }
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Store plot elements for animation
    actor_plots = {}
    trail_plots = {}
    
    # Initialize plot elements for each track
    for track_id in tracks:
        track_states = scenario_states[scenario_states['track_id'] == track_id]
        if len(track_states) == 0:
            continue
        
        object_type = track_states['object_type'].iloc[0]
        is_sdc = track_states['is_sdc'].iloc[0]
        color = colors.get(object_type, 'black')
        
        # Current position marker
        actor_plot, = ax.plot([], [], 'o', color=color, markersize=12, 
                             markeredgecolor='black', markeredgewidth=1)
        actor_plots[track_id] = actor_plot
        
        # Trail (past positions)
        trail_plot, = ax.plot([], [], '-', color=color, linewidth=1, alpha=0.3)
        trail_plots[track_id] = trail_plot
    
    # Setup plot limits
    valid_states = scenario_states[scenario_states['valid'] == True]
    if len(valid_states) > 0:
        x_min, x_max = valid_states['x'].min() - 5, valid_states['x'].max() + 5
        y_min, y_max = valid_states['y'].min() - 5, valid_states['y'].max() + 5
    else:
        x_min, x_max = -50, 50
        y_min, y_max = -50, 50
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Scenario Animation: {first_scenario_id}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend
    legend_elements = []
    for track_id in tracks:
        track_states = scenario_states[scenario_states['track_id'] == track_id]
        if len(track_states) == 0:
            continue
        
        object_type = track_states['object_type'].iloc[0]
        is_sdc = track_states['is_sdc'].iloc[0]
        color = colors.get(object_type, 'black')
        
        track_num = track_id.split('_')[-1]
        label = f"{object_type} ({'SDC' if is_sdc else track_num})"
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                        markersize=8, markeredgecolor='black', markeredgewidth=1, label=label))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def animate_frame(frame_num):
        """Animation function for each frame."""
        if frame_num >= len(timesteps):
            return []
        
        current_timestep = timesteps[frame_num]
        
        # Update time text
        time_seconds = current_timestep * 0.1  # 10Hz data
        time_text.set_text(f'Time: {time_seconds:.1f}s\nFrame: {frame_num}/{len(timesteps)}')
        
        # Update each track
        artists = [time_text]
        for track_id in tracks:
            track_states = scenario_states[scenario_states['track_id'] == track_id]
            if len(track_states) == 0:
                continue
            
            # Get states up to current timestep
            past_states = track_states[track_states['timestep'] <= current_timestep]
            current_state = track_states[track_states['timestep'] == current_timestep]
            
            # Update current position
            if len(current_state) > 0:
                x = current_state['x'].iloc[0]
                y = current_state['y'].iloc[0]
                valid = current_state['valid'].iloc[0]
                
                if valid:
                    actor_plots[track_id].set_data([x], [y])
                    actor_plots[track_id].set_markersize(12)
                    actor_plots[track_id].set_alpha(1.0)
                else:
                    # Make invalid states smaller/fainter
                    actor_plots[track_id].set_data([x], [y])
                    actor_plots[track_id].set_markersize(6)
                    actor_plots[track_id].set_alpha(0.3)
            else:
                actor_plots[track_id].set_data([], [])
            
            # Update trail (past valid positions)
            valid_past_states = past_states[past_states['valid'] == True]
            if len(valid_past_states) > 0:
                trail_x = valid_past_states['x'].values
                trail_y = valid_past_states['y'].values
                trail_plots[track_id].set_data(trail_x, trail_y)
            else:
                trail_plots[track_id].set_data([], [])
            
            artists.extend([actor_plots[track_id], trail_plots[track_id]])
        
        return artists
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(timesteps), 
                                 interval=100, blit=True, repeat=True)
    
    # Ensure exports directory exists
    exports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as GIF
    gif_file = exports_dir / 'first_scenario.gif'
    print(f"Saving animation to: {gif_file}")
    
    try:
        anim.save(gif_file, writer='pillow', fps=10, dpi=100)
        print(f"✅ Animation saved to: {gif_file}")
    except Exception as e:
        print(f"⚠️  Could not save GIF: {e}")
        print("Trying to save as MP4...")
        try:
            mp4_file = exports_dir / 'first_scenario.mp4'
            anim.save(mp4_file, writer='ffmpeg', fps=10, dpi=100)
            print(f"✅ Animation saved as MP4: {mp4_file}")
        except Exception as e2:
            print(f"⚠️  Could not save MP4 either: {e2}")
            print("Displaying animation instead...")
            plt.show()
    
    plt.close()
    
    # Print summary
    object_counts = scenario_states['object_type'].value_counts()
    print(f"📊 Object type distribution: {dict(object_counts)}")
    print(f"📊 Animation frames: {len(timesteps)}")
    print(f"📊 Duration: {len(timesteps) * 0.1:.1f} seconds")

def main():
    print("=" * 60)
    print("WAYMO VALIDATION LAB - ANIMATE FIRST SCENARIO")
    print("=" * 60)
    
    try:
        animate_scenario()
        
        print("=" * 60)
        print("ANIMATION COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Animation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
