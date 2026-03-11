#!/usr/bin/env python3
"""
Show Scenarios Overview

Quick script to display scenario information.
"""

import pandas as pd
import sys
from pathlib import Path

def main():
    # Load scenarios
    scenarios_file = Path('data/silver/scenarios.parquet')
    if not scenarios_file.exists():
        print('❌ Scenarios file not found!')
        print('Please run the parser first:')
        print('python scripts/real_protobuf_parser.py')
        sys.exit(1)

    scenarios_df = pd.read_parquet(scenarios_file)
    print('=' * 60)
    print('WAYMO SCENARIOS OVERVIEW')
    print('=' * 60)
    print(f'Total scenarios: {len(scenarios_df)}')
    print()

    # Show scenarios table
    print('SCENARIOS TABLE:')
    print('-' * 50)
    display_cols = ['scenario_id', 'data_source', 'num_tracks', 'num_steps']
    print(scenarios_df[display_cols].to_string(index=False))
    print()

    # Show data sources
    if 'data_source' in scenarios_df.columns:
        sources = scenarios_df['data_source'].value_counts()
        print('DATA SOURCES:')
        print('-' * 20)
        for source, count in sources.items():
            print(f'{source}: {count}')
        print()

    # Show statistics
    print('SCENARIO STATISTICS:')
    print('-' * 25)
    avg_tracks = scenarios_df['num_tracks'].mean()
    total_tracks = scenarios_df['num_tracks'].sum()
    total_steps = scenarios_df['num_steps'].iloc[0]
    total_states = total_tracks * total_steps

    print(f'Average tracks per scenario: {avg_tracks:.1f}')
    print(f'Total tracks across all scenarios: {total_tracks}')
    print(f'Total timesteps per scenario: {total_steps}')
    print(f'Total state records: {total_states}')
    print()

    # Load tracks and states for more details
    try:
        tracks_df = pd.read_parquet('data/silver/tracks.parquet')
        states_df = pd.read_parquet('data/silver/states.parquet')

        print('TRACK BREAKDOWN:')
        print('-' * 20)
        obj_counts = tracks_df['object_type'].value_counts()
        for obj_type, count in obj_counts.items():
            print(f'{obj_type}: {count}')
        print()

        sdc_tracks = tracks_df[tracks_df['is_sdc'] == True]
        print(f'SDC tracks: {len(sdc_tracks)}')
        print(f'States loaded: {len(states_df)}')
        print()

    except Exception as e:
        print(f'Could not load tracks/states: {e}')

if __name__ == "__main__":
    main()
