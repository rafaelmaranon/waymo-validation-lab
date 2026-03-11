#!/usr/bin/env python3
"""
Compute real risk metrics from scenario trajectories.

Uses Time-to-Collision (TTC) and closing speed to calculate a composite
risk score for each scenario. Replaces proxy risk with physics-based metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# ---------- paths ----------
project_root = Path(__file__).parent.parent
SILVER_DIR = project_root / "data" / "silver"
GOLD_DIR = project_root / "data" / "gold"

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


def compute_scenario_risk(scenario_id: str, states_df: pd.DataFrame, tracks_df: pd.DataFrame) -> dict:
    """Compute risk metrics for a single scenario."""
    
    # Get SDC track
    scenario_tracks = tracks_df[tracks_df['scenario_id'] == scenario_id]
    sdc_track = scenario_tracks[scenario_tracks['is_sdc'] == True]
    
    # Edge case: no SDC found
    if len(sdc_track) == 0:
        return {
            'scenario_id': scenario_id,
            'risk_score': 0.0,
            'min_ttc_s': None,
            'max_closing_speed_mps': 0.0,
            'num_ttc_below_3s': 0,
            'num_ttc_below_1_5s': 0,
            'closest_risk_actor_track_id': None,
            'closest_risk_actor_type': None,
            'min_risk_distance_m': None,
            'risk_score_components': json.dumps({
                'ttc_component': 0.0,
                'closing_component': 0.0,
                'breach_component': 0.0,
                'note': 'No SDC track found'
            })
        }
    
    sdc_track_id = sdc_track.iloc[0]['track_id']
    
    # Get valid states for this scenario
    scenario_states = states_df[
        (states_df['scenario_id'] == scenario_id) & 
        (states_df['valid'] == True)
    ].copy()
    
    # Split SDC and other actors
    sdc_states = scenario_states[scenario_states['track_id'] == sdc_track_id].copy()
    other_states = scenario_states[scenario_states['track_id'] != sdc_track_id].copy()
    
    if len(sdc_states) == 0 or len(other_states) == 0:
        return {
            'scenario_id': scenario_id,
            'risk_score': 0.0,
            'min_ttc_s': None,
            'max_closing_speed_mps': 0.0,
            'num_ttc_below_3s': 0,
            'num_ttc_below_1_5s': 0,
            'closest_risk_actor_track_id': None,
            'closest_risk_actor_type': None,
            'min_risk_distance_m': None,
            'risk_score_components': json.dumps({
                'ttc_component': 0.0,
                'closing_component': 0.0,
                'breach_component': 0.0,
                'note': 'No valid states for SDC or other actors'
            })
        }
    
    # Initialize risk metrics
    all_ttc_values = []
    all_closing_speeds = []
    all_distances = []
    event_details = []  # (ttc, closing_speed, distance, actor_id, actor_type)
    
    # Process each timestep
    timesteps = sorted(sdc_states['timestep'].unique())
    
    for timestep in timesteps:
        sdc_at_t = sdc_states[sdc_states['timestep'] == timestep]
        others_at_t = other_states[other_states['timestep'] == timestep]
        
        if len(sdc_at_t) == 0 or len(others_at_t) == 0:
            continue
        
        sdc_pos = sdc_at_t.iloc[0]
        sdc_x, sdc_y = sdc_pos['x'], sdc_pos['y']
        sdc_vx, sdc_vy = sdc_pos['velocity_x'], sdc_pos['velocity_y']
        
        # Compute TTC and closing speed for each other actor
        for _, actor in others_at_t.iterrows():
            # Relative position and velocity
            dx = actor['x'] - sdc_x
            dy = actor['y'] - sdc_y
            distance = np.sqrt(dx**2 + dy**2)
            
            dvx = actor['velocity_x'] - sdc_vx
            dvy = actor['velocity_y'] - sdc_vy
            
            # Closing speed (positive when approaching)
            closing_speed = -(dx * dvx + dy * dvy) / max(distance, 1e-6)
            
            # Only consider closing scenarios
            if closing_speed > 0:
                ttc = distance / closing_speed
                all_ttc_values.append(ttc)
                all_closing_speeds.append(closing_speed)
                all_distances.append(distance)
                
                event_details.append({
                    'ttc': ttc,
                    'closing_speed': closing_speed,
                    'distance': distance,
                    'actor_id': actor['track_id'],
                    'actor_type': actor['object_type']
                })
    
    # Aggregate metrics
    if all_ttc_values:
        min_ttc = min(all_ttc_values)
        max_closing_speed = max(all_closing_speeds)
        num_ttc_below_3s   = sum(1 for ttc in all_ttc_values if ttc < 3.0)
        num_ttc_below_1_5s = sum(1 for ttc in all_ttc_values if ttc < 1.5)
        num_ttc_below_2s   = sum(1 for ttc in all_ttc_values if ttc < 2.0)
        num_ttc_below_1s   = sum(1 for ttc in all_ttc_values if ttc < 1.0)
        
        # Find the highest risk event (lowest TTC)
        highest_risk_event = min(event_details, key=lambda e: e['ttc'])
        closest_risk_actor_track_id = highest_risk_event['actor_id']
        closest_risk_actor_type = highest_risk_event['actor_type']
        min_risk_distance = highest_risk_event['distance']
    else:
        min_ttc = None
        max_closing_speed = 0.0
        num_ttc_below_3s = 0
        num_ttc_below_1_5s = 0
        closest_risk_actor_track_id = None
        closest_risk_actor_type = None
        min_risk_distance = None
        num_ttc_below_2s   = 0
        num_ttc_below_1s   = 0

    # Compute risk score components (recalibrated — normal driving = low score)
    ttc_warning  = 2.0
    ttc_critical = 0.8
    if min_ttc is not None:
        ttc_component = max(0.0, min(1.0,
            (ttc_warning - min_ttc) / (ttc_warning - ttc_critical)
        ))
    else:
        ttc_component = 0.0

    closing_component  = min(1.0, max_closing_speed / 40.0)
    exposure_component = min(1.0, (num_ttc_below_2s + 2 * num_ttc_below_1s) / 150.0)

    # Composite risk score with power transform
    risk_score = (
        0.50 * ttc_component +
        0.30 * closing_component +
        0.20 * exposure_component
    ) ** 1.5
    
    # Store components for debugging
    components = {
        'ttc_component': ttc_component,
        'closing_component': closing_component,
        'exposure_component': exposure_component,
        'min_ttc_s': min_ttc,
        'max_closing_speed_mps': max_closing_speed,
        'num_ttc_below_3s': num_ttc_below_3s,
        'num_ttc_below_1_5s': num_ttc_below_1_5s,
        'num_ttc_below_2s': num_ttc_below_2s,
        'num_ttc_below_1s': num_ttc_below_1s,
    }

    return {
        'scenario_id': scenario_id,
        'risk_score': risk_score,
        'min_ttc_s': min_ttc,
        'max_closing_speed_mps': max_closing_speed,
        'num_ttc_below_3s': num_ttc_below_3s,
        'num_ttc_below_1_5s': num_ttc_below_1_5s,
        'num_ttc_below_2s': num_ttc_below_2s,
        'num_ttc_below_1s': num_ttc_below_1s,
        'closest_risk_actor_track_id': closest_risk_actor_track_id,
        'closest_risk_actor_type': closest_risk_actor_type,
        'min_risk_distance_m': min_risk_distance,
        'risk_score_components': json.dumps(components)
    }


def main():
    print("=" * 70)
    print("WAYMO VALIDATION LAB — COMPUTE RISK METRICS")
    print("=" * 70)
    
    # Load data
    scenarios_df, tracks_df, states_df = load_silver()
    
    # Process each scenario
    results = []
    scenario_ids = scenarios_df['scenario_id'].tolist()
    
    for i, scenario_id in enumerate(scenario_ids, 1):
        metrics = compute_scenario_risk(scenario_id, states_df, tracks_df)
        results.append(metrics)
        
        min_ttc_str = f"{metrics['min_ttc_s']:.2f}s" if metrics['min_ttc_s'] else "N/A"
        print(f"[{i:2d}/{len(scenario_ids)}] scenario_id={scenario_id}  "
              f"min_ttc={min_ttc_str}  "
              f"max_closing={metrics['max_closing_speed_mps']:.1f}m/s  "
              f"risk_score={metrics['risk_score']:.2f}")
    
    # Create and save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('risk_score', ascending=False)
    
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(GOLD_DIR / 'risk_metrics.parquet', index=False)
    
    print("\n" + "=" * 70)
    print("TOP 5 SCENARIOS BY RISK SCORE")
    print("=" * 70)
    top_5 = results_df.head(5)[[
        'scenario_id', 'risk_score', 'min_ttc_s', 'max_closing_speed_mps', 
        'num_ttc_below_3s', 'num_ttc_below_1_5s'
    ]]
    print(top_5.to_string(index=False))
    
    print(f"\n✅ Risk metrics saved to data/gold/risk_metrics.parquet")
    print("=" * 70)


if __name__ == "__main__":
    main()
