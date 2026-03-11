#!/usr/bin/env python3
"""
View Scenarios

Interactive viewer for Waymo scenario data.
Shows scenarios in multiple formats.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def show_scenarios_overview():
    """Show overview of all scenarios."""
    
    print("=" * 60)
    print("WAYMO SCENARIOS OVERVIEW")
    print("=" * 60)
    
    # Load scenarios
    scenarios_file = project_root / 'data' / 'silver' / 'scenarios.parquet'
    if not scenarios_file.exists():
        print(f"❌ Scenarios file not found: {scenarios_file}")
        print("Please run the parser first:")
        print("python scripts/real_protobuf_parser.py")
        return
    
    scenarios_df = pd.read_parquet(scenarios_file)
    print(f"Total scenarios: {len(scenarios_df)}")
    print()
    
    # Show scenarios table
    print("SCENARIOS TABLE:")
    print("-" * 40)
    display_cols = ['scenario_id', 'data_source', 'num_tracks', 'num_steps']
    print(scenarios_df[display_cols].to_string(index=False))
    print()
    
    # Show data sources
    if 'data_source' in scenarios_df.columns:
        sources = scenarios_df['data_source'].value_counts()
        print("DATA SOURCES:")
        print("-" * 20)
        for source, count in sources.items():
            print(f"{source}: {count}")
        print()
    
    return scenarios_df

def show_scenario_details(scenario_id: str):
    """Show detailed information about a specific scenario."""
    
    print("=" * 60)
    print(f"SCENARIO DETAILS: {scenario_id}")
    print("=" * 60)
    
    # Load states
    states_file = project_root / 'data' / 'silver' / 'states.parquet'
    tracks_file = project_root / 'data' / 'silver' / 'tracks.parquet'
    
    states_df = pd.read_parquet(states_file)
    tracks_df = pd.read_parquet(tracks_file)
    
    # Filter for this scenario
    scenario_states = states_df[states_df['scenario_id'] == scenario_id]
    scenario_tracks = tracks_df[tracks_df['scenario_id'] == scenario_id]
    
    print(f"Scenario: {scenario_id}")
    print(f"Tracks: {len(scenario_tracks)}")
    print(f"States: {len(scenario_states)}")
    print()
    
    # Show track summary
    print("TRACKS SUMMARY:")
    print("-" * 30)
    track_summary = scenario_tracks[['track_id', 'object_type', 'is_sdc', 'states_count']]
    print(track_summary.to_string(index=False))
    print()
    
    # Show object type distribution
    obj_counts = scenario_tracks['object_type'].value_counts()
    print("OBJECT TYPE DISTRIBUTION:")
    print("-" * 25)
    for obj_type, count in obj_counts.items():
        print(f"{obj_type}: {count}")
    print()
    
    # Show spatial bounds
    valid_states = scenario_states[scenario_states['valid'] == True]
    if len(valid_states) > 0:
        print("SPATIAL BOUNDS:")
        print("-" * 20)
        print(f"X range: {valid_states['x'].min():.2f} to {valid_states['x'].max():.2f} m")
        print(f"Y range: {valid_states['y'].min():.2f} to {valid_states['y'].max():.2f} m")
        print(f"Z range: {valid_states['z'].min():.2f} to {valid_states['z'].max():.2f} m")
        print()
        
        # Show sample states
        print("SAMPLE STATES (first 5):")
        print("-" * 40)
        sample_cols = ['track_id', 'timestep', 'x', 'y', 'velocity_x', 'velocity_y', 'valid']
        print(scenario_states[sample_cols].head().to_string(index=False))
        print()

def show_available_files():
    """Show available output files."""
    
    print("=" * 60)
    print("AVAILABLE OUTPUT FILES")
    print("=" * 60)
    
    # Check parquet files
    silver_dir = project_root / 'data' / 'silver'
    gold_dir = project_root / 'data' / 'gold'
    exports_dir = project_root / 'data' / 'exports'
    
    print("PARQUET FILES:")
    print("-" * 20)
    for file_path in [silver_dir / 'scenarios.parquet', 
                      silver_dir / 'tracks.parquet',
                      silver_dir / 'states.parquet',
                      gold_dir / 'scenario_metrics.parquet']:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file_path.name}: {size_mb:.1f} MB")
        else:
            print(f"❌ {file_path.name}: NOT FOUND")
    print()
    
    print("VISUALIZATION FILES:")
    print("-" * 25)
    viz_files = [
        exports_dir / 'first_scenario_plot.png',
        exports_dir / 'first_scenario.gif'
    ]
    for file_path in viz_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file_path.name}: {size_mb:.1f} MB")
        else:
            print(f"❌ {file_path.name}: NOT FOUND")
    print()
    
    # Check JSON files
    json_dir = exports_dir / 'scenario_json'
    if json_dir.exists():
        json_files = list(json_dir.glob('*.json'))
        print(f"JSON FILES: {len(json_files)} files")
        print(f"Directory: {json_dir}")
        if json_files:
            print("Sample files:")
            for file_path in json_files[:3]:
                print(f"  - {file_path.name}")
    print()

def interactive_viewer():
    """Interactive scenario viewer."""
    
    while True:
        print("\n" + "=" * 60)
        print("WAYMO SCENARIO VIEWER")
        print("=" * 60)
        print("1. Show scenarios overview")
        print("2. Show scenario details")
        print("3. Show available files")
        print("4. Exit")
        print()
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            scenarios_df = show_scenarios_overview()
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            scenarios_df = pd.read_parquet(project_root / 'data' / 'silver' / 'scenarios.parquet')
            scenario_ids = scenarios_df['scenario_id'].tolist()
            
            print("\nAvailable scenarios:")
            for i, sid in enumerate(scenario_ids, 1):
                print(f"{i}. {sid}")
            
            try:
                choice_num = int(input(f"\nEnter scenario number (1-{len(scenario_ids)}): "))
                if 1 <= choice_num <= len(scenario_ids):
                    scenario_id = scenario_ids[choice_num - 1]
                    show_scenario_details(scenario_id)
                else:
                    print("Invalid choice!")
            except ValueError:
                print("Invalid input!")
            
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            show_available_files()
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice! Please try again.")
            input("Press Enter to continue...")

def main():
    print("WAYMO VALIDATION LAB - SCENARIO VIEWER")
    print("This tool helps you explore the parsed Waymo scenarios.")
    print()
    
    # Check if data exists
    scenarios_file = project_root / 'data' / 'silver' / 'scenarios.parquet'
    if not scenarios_file.exists():
        print("❌ No scenario data found!")
        print("Please run the parser first:")
        print("python scripts/real_protobuf_parser.py")
        return
    
    interactive_viewer()

if __name__ == "__main__":
    main()
