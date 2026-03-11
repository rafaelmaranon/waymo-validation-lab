#!/usr/bin/env python3
"""
Compute SDC-centric interaction metrics for each scenario.

Reads silver tables, computes pairwise distances between SDC and other actors,
and outputs interaction metrics with a composite ranking score.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- constants ----------
CLOSE_THRESHOLD_M = 5.0   # meters — defines "close interaction"

project_root = Path(__file__).parent.parent
SILVER_DIR = project_root / 'data' / 'silver'
GOLD_DIR   = project_root / 'data' / 'gold'

# ---------- functions ----------
def load_silver() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all silver tables."""
    print("Loading silver tables...")
    
    scenarios_df = pd.read_parquet(SILVER_DIR / 'scenarios.parquet')
    tracks_df = pd.read_parquet(SILVER_DIR / 'tracks.parquet')
    states_df = pd.read_parquet(SILVER_DIR / 'states.parquet')
    
    print(f"  scenarios: {len(scenarios_df)} rows")
    print(f"  tracks: {len(tracks_df)} rows")
    print(f"  states: {len(states_df)} rows")
    
    return scenarios_df, tracks_df, states_df


def compute_scenario_interactions(scenario_id: str, states_df: pd.DataFrame, tracks_df: pd.DataFrame) -> dict:
    """Compute interaction metrics for a single scenario."""
    
    # Filter to this scenario and valid states only
    scenario_states = states_df[
        (states_df['scenario_id'] == scenario_id) & 
        (states_df['valid'] == True)
    ].copy()
    
    # Get SDC track info
    scenario_tracks = tracks_df[tracks_df['scenario_id'] == scenario_id]
    sdc_track = scenario_tracks[scenario_tracks['is_sdc'] == True]
    
    # Edge case: no SDC found
    if len(sdc_track) == 0:
        return {
            'scenario_id': scenario_id,
            'min_sdc_distance_m': 0.0,
            'mean_min_sdc_distance_m': 0.0,
            'num_close_interactions': 0,
            'num_timesteps_with_close_actor': 0,
            'closest_actor_type': None,
            'closest_actor_track_id': None,
            'sdc_avg_speed_mps': 0.0,
            'sdc_max_speed_mps': 0.0,
            'sdc_distance_traveled_m': 0.0,
            'num_unique_close_actors': 0,
            'scenario_interest_score': 0.0,
        }
    
    sdc_track_id = sdc_track.iloc[0]['track_id']
    
    # Split SDC and other actor states
    sdc_states = scenario_states[scenario_states['track_id'] == sdc_track_id].copy()
    other_states = scenario_states[scenario_states['track_id'] != sdc_track_id].copy()
    
    if len(sdc_states) == 0:
        return {
            'scenario_id': scenario_id,
            'min_sdc_distance_m': 0.0,
            'mean_min_sdc_distance_m': 0.0,
            'num_close_interactions': 0,
            'num_timesteps_with_close_actor': 0,
            'closest_actor_type': None,
            'closest_actor_track_id': None,
            'sdc_avg_speed_mps': 0.0,
            'sdc_max_speed_mps': 0.0,
            'sdc_distance_traveled_m': 0.0,
            'num_unique_close_actors': 0,
            'scenario_interest_score': 0.0,
        }
    
    # Compute SDC speed
    sdc_states['speed'] = np.sqrt(sdc_states['velocity_x']**2 + sdc_states['velocity_y']**2)
    sdc_avg_speed = sdc_states['speed'].mean()
    sdc_max_speed = sdc_states['speed'].max()
    
    # Compute SDC distance traveled
    sdc_states_sorted = sdc_states.sort_values('timestep')
    positions = sdc_states_sorted[['x', 'y']].values
    if len(positions) > 1:
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        sdc_distance_traveled = distances.sum()
    else:
        sdc_distance_traveled = 0.0
    
    # Initialize metrics
    min_distances = []
    close_interactions_count = 0
    timesteps_with_close_actor = 0
    close_actors = set()
    
    # Process each timestep
    timesteps = sorted(sdc_states['timestep'].unique())
    
    for timestep in timesteps:
        sdc_at_t = sdc_states[sdc_states['timestep'] == timestep]
        others_at_t = other_states[other_states['timestep'] == timestep]
        
        if len(sdc_at_t) == 0 or len(others_at_t) == 0:
            continue
        
        sdc_pos = sdc_at_t.iloc[0]
        sdc_x, sdc_y = sdc_pos['x'], sdc_pos['y']
        
        # Compute distances to all other actors at this timestep
        distances = np.sqrt((others_at_t['x'] - sdc_x)**2 + (others_at_t['y'] - sdc_y)**2)
        
        # Track close interactions
        close_mask = distances < CLOSE_THRESHOLD_M
        close_count = close_mask.sum()
        
        if close_count > 0:
            close_interactions_count += close_count
            timesteps_with_close_actor += 1
            close_actors.update(others_at_t[close_mask]['track_id'].tolist())
        
        # Record minimum distance at this timestep
        if len(distances) > 0:
            min_distances.append(distances.min())
    
    # Aggregate metrics
    if min_distances:
        min_sdc_distance = min(min_distances)
        mean_min_sdc_distance = np.mean(min_distances)
    else:
        min_sdc_distance = 0.0
        mean_min_sdc_distance = 0.0
    
    num_unique_close_actors = len(close_actors)
    
    # Find closest actor details
    if min_distances:
        # Find the timestep and actor that achieved the minimum distance
        min_dist = min_sdc_distance
        closest_actor_info = None
        
        for timestep in timesteps:
            sdc_at_t = sdc_states[sdc_states['timestep'] == timestep]
            others_at_t = other_states[other_states['timestep'] == timestep]
            
            if len(sdc_at_t) == 0 or len(others_at_t) == 0:
                continue
            
            sdc_pos = sdc_at_t.iloc[0]
            sdc_x, sdc_y = sdc_pos['x'], sdc_pos['y']
            distances = np.sqrt((others_at_t['x'] - sdc_x)**2 + (others_at_t['y'] - sdc_y)**2)
            
            if len(distances) > 0 and distances.min() == min_dist:
                closest_idx = distances.idxmin()
                closest_actor = others_at_t.loc[closest_idx]
                closest_actor_info = {
                    'type': closest_actor['object_type'],
                    'track_id': closest_actor['track_id']
                }
                break
        
        if closest_actor_info:
            closest_actor_type = closest_actor_info['type']
            closest_actor_track_id = closest_actor_info['track_id']
        else:
            closest_actor_type = None
            closest_actor_track_id = None
    else:
        closest_actor_type = None
        closest_actor_track_id = None
    
    # Compute composite interest score
    score = (
        0.30 * min(1.0, 10.0 / (min_sdc_distance + 0.1))     # closer = more interesting
      + 0.25 * min(1.0, close_interactions_count / 50.0)      # more close interactions = more interesting
      + 0.20 * min(1.0, sdc_max_speed / 25.0)               # faster SDC = more interesting
      + 0.15 * min(1.0, num_unique_close_actors / 10.0)      # more actors nearby = more interesting
      + 0.10 * min(1.0, sdc_distance_traveled / 200.0)       # more movement = more interesting
    )
    
    return {
        'scenario_id': scenario_id,
        'min_sdc_distance_m': min_sdc_distance,
        'mean_min_sdc_distance_m': mean_min_sdc_distance,
        'num_close_interactions': close_interactions_count,
        'num_timesteps_with_close_actor': timesteps_with_close_actor,
        'closest_actor_type': closest_actor_type,
        'closest_actor_track_id': closest_actor_track_id,
        'sdc_avg_speed_mps': sdc_avg_speed,
        'sdc_max_speed_mps': sdc_max_speed,
        'sdc_distance_traveled_m': sdc_distance_traveled,
        'num_unique_close_actors': num_unique_close_actors,
        'scenario_interest_score': score,
    }


def main():
    print("=" * 70)
    print("WAYMO VALIDATION LAB — INTERACTION METRICS")
    print("=" * 70)
    
    # Load data
    scenarios_df, tracks_df, states_df = load_silver()
    
    # Process each scenario
    results = []
    scenario_ids = scenarios_df['scenario_id'].tolist()
    
    for i, scenario_id in enumerate(scenario_ids, 1):
        metrics = compute_scenario_interactions(scenario_id, states_df, tracks_df)
        results.append(metrics)
        
        print(f"[{i:2d}/{len(scenario_ids)}] scenario_id={scenario_id}  "
              f"min_dist={metrics['min_sdc_distance_m']:.2f}m  "
              f"close_interactions={metrics['num_close_interactions']}  "
              f"score={metrics['scenario_interest_score']:.2f}")
    
    # Create and save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('scenario_interest_score', ascending=False)
    
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(GOLD_DIR / 'interaction_metrics.parquet', index=False)
    
    print("\n" + "=" * 70)
    print("TOP 5 SCENARIOS BY INTEREST SCORE")
    print("=" * 70)
    top_5 = results_df.head(5)[['scenario_id', 'min_sdc_distance_m', 'num_close_interactions', 'scenario_interest_score']]
    print(top_5.to_string(index=False))
    
    print(f"\n✅ Interaction metrics saved to data/gold/interaction_metrics.parquet")
    print("=" * 70)


if __name__ == "__main__":
    main()
