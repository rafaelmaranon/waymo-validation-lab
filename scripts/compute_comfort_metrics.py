#!/usr/bin/env python3
"""
Compute comfort metrics from SDC motion patterns.

Uses acceleration, jerk, and heading rate to calculate a comfort score
for each scenario. Higher scores indicate less comfortable motion.
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

# ---------- constants ----------
DT = 0.1  # Waymo scenario timestep: 0.1 seconds (10 Hz)

def wrapped_angle_diff(angle1: float, angle2: float) -> float:
    """Compute the smallest difference between two angles, handling wrap-around."""
    diff = angle1 - angle2
    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff

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


def compute_scenario_comfort(scenario_id: str, states_df: pd.DataFrame, tracks_df: pd.DataFrame) -> dict:
    """Compute comfort metrics for a single scenario."""
    
    # Get SDC track
    scenario_tracks = tracks_df[tracks_df['scenario_id'] == scenario_id]
    sdc_track = scenario_tracks[scenario_tracks['is_sdc'] == True]
    
    # Edge case: no SDC found
    if len(sdc_track) == 0:
        return {
            'scenario_id': scenario_id,
            'max_acceleration_mps2': 0.0,
            'max_deceleration_mps2': 0.0,
            'max_jerk_mps3': 0.0,
            'mean_abs_jerk_mps3': 0.0,
            'max_heading_rate_radps': 0.0,
            'comfort_score': 0.0,
            'comfort_score_components': json.dumps({
                'accel_component': 0.0,
                'decel_component': 0.0,
                'jerk_component': 0.0,
                'heading_component': 0.0,
                'note': 'No SDC track found'
            })
        }
    
    sdc_track_id = sdc_track.iloc[0]['track_id']
    
    # Get valid SDC states
    scenario_states = states_df[
        (states_df['scenario_id'] == scenario_id) & 
        (states_df['valid'] == True) &
        (states_df['track_id'] == sdc_track_id)
    ].copy()
    
    if len(scenario_states) < 2:
        return {
            'scenario_id': scenario_id,
            'max_acceleration_mps2': 0.0,
            'max_deceleration_mps2': 0.0,
            'max_jerk_mps3': 0.0,
            'mean_abs_jerk_mps3': 0.0,
            'max_heading_rate_radps': 0.0,
            'comfort_score': 0.0,
            'comfort_score_components': json.dumps({
                'accel_component': 0.0,
                'decel_component': 0.0,
                'jerk_component': 0.0,
                'heading_component': 0.0,
                'note': 'Insufficient SDC states'
            })
        }
    
    # Sort by timestep
    sdc_states = scenario_states.sort_values('timestep')
    
    # Compute speed magnitude
    sdc_states['speed_mps'] = np.sqrt(sdc_states['velocity_x']**2 + sdc_states['velocity_y']**2)
    
    # Compute acceleration (longitudinal speed change)
    sdc_states['acceleration_mps2'] = sdc_states['speed_mps'].diff() / DT
    
    # Compute jerk (change in acceleration)
    sdc_states['jerk_mps3'] = sdc_states['acceleration_mps2'].diff() / DT
    
    # Compute heading rate
    sdc_states['heading_rate_radps'] = sdc_states['heading'].diff().apply(
        lambda x: wrapped_angle_diff(x, 0) / DT if not pd.isna(x) else 0.0
    )
    
    # Remove NaN values from first row
    valid_metrics = sdc_states.dropna(subset=['acceleration_mps2', 'jerk_mps3', 'heading_rate_radps'])
    
    if len(valid_metrics) == 0:
        return {
            'scenario_id': scenario_id,
            'max_acceleration_mps2': 0.0,
            'max_deceleration_mps2': 0.0,
            'max_jerk_mps3': 0.0,
            'mean_abs_jerk_mps3': 0.0,
            'max_heading_rate_radps': 0.0,
            'comfort_score': 0.0,
            'comfort_score_components': json.dumps({
                'accel_component': 0.0,
                'decel_component': 0.0,
                'jerk_component': 0.0,
                'heading_component': 0.0,
                'note': 'No valid metrics after processing'
            })
        }
    
    # Extract metrics
    max_acceleration = valid_metrics['acceleration_mps2'].max()
    max_deceleration = abs(valid_metrics['acceleration_mps2'].min())
    max_jerk = valid_metrics['jerk_mps3'].abs().max()
    mean_abs_jerk = valid_metrics['jerk_mps3'].abs().mean()
    max_heading_rate = valid_metrics['heading_rate_radps'].abs().max()
    
    # Compute comfort score components
    accel_component = min(1.0, max_acceleration / 4.0)
    decel_component = min(1.0, max_deceleration / 4.0)
    jerk_component = min(1.0, max_jerk / 10.0)
    heading_component = min(1.0, max_heading_rate / 0.8)
    
    # Composite comfort score (higher = less comfortable)
    comfort_score = (
        0.25 * accel_component +
        0.30 * decel_component +
        0.30 * jerk_component +
        0.15 * heading_component
    )
    
    # Store components for debugging
    components = {
        'accel_component': accel_component,
        'decel_component': decel_component,
        'jerk_component': jerk_component,
        'heading_component': heading_component,
        'max_acceleration_mps2': max_acceleration,
        'max_deceleration_mps2': max_deceleration,
        'max_jerk_mps3': max_jerk,
        'max_heading_rate_radps': max_heading_rate
    }
    
    return {
        'scenario_id': scenario_id,
        'max_acceleration_mps2': max_acceleration,
        'max_deceleration_mps2': max_deceleration,
        'max_jerk_mps3': max_jerk,
        'mean_abs_jerk_mps3': mean_abs_jerk,
        'max_heading_rate_radps': max_heading_rate,
        'comfort_score': comfort_score,
        'comfort_score_components': json.dumps(components)
    }


def main():
    print("=" * 70)
    print("WAYMO VALIDATION LAB — COMPUTE COMFORT METRICS")
    print("=" * 70)
    
    # Load data
    scenarios_df, tracks_df, states_df = load_silver()
    
    # Process each scenario
    results = []
    scenario_ids = scenarios_df['scenario_id'].tolist()
    
    for i, scenario_id in enumerate(scenario_ids, 1):
        metrics = compute_scenario_comfort(scenario_id, states_df, tracks_df)
        results.append(metrics)
        
        print(f"[{i:2d}/{len(scenario_ids)}] scenario_id={scenario_id}  "
              f"accel={metrics['max_acceleration_mps2']:.2f}  "
              f"decel={metrics['max_deceleration_mps2']:.2f}  "
              f"jerk={metrics['max_jerk_mps3']:.2f}  "
              f"comfort={metrics['comfort_score']:.2f}")
    
    # Create and save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('comfort_score', ascending=False)
    
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(GOLD_DIR / 'comfort_metrics.parquet', index=False)
    
    print("\n" + "=" * 70)
    print("TOP 5 SCENARIOS BY COMFORT SCORE (least comfortable)")
    print("=" * 70)
    top_5 = results_df.head(5)[[
        'scenario_id', 'comfort_score', 'max_acceleration_mps2', 
        'max_deceleration_mps2', 'max_jerk_mps3', 'max_heading_rate_radps'
    ]]
    print(top_5.to_string(index=False))
    
    print(f"\n✅ Comfort metrics saved to data/gold/comfort_metrics.parquet")
    print("=" * 70)


if __name__ == "__main__":
    main()
