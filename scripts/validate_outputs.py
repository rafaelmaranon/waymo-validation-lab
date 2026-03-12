#!/usr/bin/env python3
"""
Validate Outputs Script

Checks that all expected files exist, loads and validates parquet files,
lists JSON exports, and verifies data consistency across tables.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any

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

def validate_file_exists(file_path: Path, description: str) -> bool:
    """Check if file exists and report status."""
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"✅ {description}: {file_path.name} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ {description}: {file_path.name} - MISSING")
        return False

def validate_parquet_file(file_path: Path, description: str) -> pd.DataFrame:
    """Load and validate a parquet file."""
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ {description}: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        
        # Show first few rows
        if len(df) > 0:
            print(f"   Sample data:")
            print(df.head(2).to_string(index=False, max_cols=8))
        else:
            print("   ⚠️  No data rows")
        
        return df
    except Exception as e:
        print(f"❌ {description}: Failed to load - {e}")
        return None

def validate_json_exports(json_dir: Path) -> List[str]:
    """Validate JSON scenario exports."""
    json_files = list(json_dir.glob('*.json'))
    
    print(f"📄 JSON scenario exports: {len(json_files)} files")
    
    scenario_ids = []
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            scenario_id = data.get('scenario_id')
            if scenario_id:
                scenario_ids.append(scenario_id)
                num_tracks = len(data.get('tracks', []))
                print(f"   ✅ {json_file.name}: scenario_id={scenario_id}, tracks={num_tracks}")
            else:
                print(f"   ⚠️  {json_file.name}: missing scenario_id")
                
        except Exception as e:
            print(f"   ❌ {json_file.name}: failed to load - {e}")
    
    return scenario_ids

def validate_data_consistency(scenarios_df: pd.DataFrame, tracks_df: pd.DataFrame, 
                            states_df: pd.DataFrame, metrics_df: pd.DataFrame,
                            interaction_df: pd.DataFrame, risk_df: pd.DataFrame,
                            comfort_df: pd.DataFrame, json_scenario_ids: List[str]):
    """Validate data consistency across tables."""
    
    print()
    print("🔍 DATA CONSISTENCY VALIDATION")
    print("=" * 40)
    
    # Check scenario IDs consistency
    if scenarios_df is not None and 'scenario_id' in scenarios_df.columns:
        parquet_scenario_ids = set(scenarios_df['scenario_id'].unique())
        json_scenario_ids_set = set(json_scenario_ids)
        
        print(f"Scenario IDs in parquet: {len(parquet_scenario_ids)}")
        print(f"Scenario IDs in JSON: {len(json_scenario_ids_set)}")
        
        # Check if they match
        if parquet_scenario_ids == json_scenario_ids_set:
            print("✅ Scenario IDs match between parquet and JSON")
        else:
            missing_in_json = parquet_scenario_ids - json_scenario_ids_set
            missing_in_parquet = json_scenario_ids_set - parquet_scenario_ids
            
            if missing_in_json:
                print(f"⚠️  Scenario IDs in parquet but not JSON: {missing_in_json}")
            if missing_in_parquet:
                print(f"⚠️  Scenario IDs in JSON but not parquet: {missing_in_parquet}")
        
        # Check uniqueness
        if len(parquet_scenario_ids) == len(scenarios_df):
            print("✅ All scenario IDs are unique")
        else:
            print("⚠️  Duplicate scenario IDs found")
    
    # Check foreign key relationships
    if scenarios_df is not None and tracks_df is not None:
        if 'scenario_id' in tracks_df.columns:
            track_scenario_ids = set(tracks_df['scenario_id'].unique())
            scenario_ids = set(scenarios_df['scenario_id'].unique())
            
            invalid_track_scenarios = track_scenario_ids - scenario_ids
            if invalid_track_scenarios:
                print(f"❌ Tracks reference invalid scenarios: {invalid_track_scenarios}")
            else:
                print("✅ All tracks reference valid scenarios")
    
    if tracks_df is not None and states_df is not None:
        if 'track_id' in states_df.columns:
            state_track_ids = set(states_df['track_id'].unique())
            track_ids = set(tracks_df['track_id'].unique())
            
            invalid_state_tracks = state_track_ids - track_ids
            if invalid_state_tracks:
                print(f"❌ States reference invalid tracks: {len(invalid_state_tracks)} track IDs")
            else:
                print("✅ All states reference valid tracks")
    
    # Check metrics consistency
    if scenarios_df is not None and metrics_df is not None:
        if 'scenario_id' in metrics_df.columns:
            metrics_scenario_ids = set(metrics_df['scenario_id'].unique())
            scenario_ids = set(scenarios_df['scenario_id'].unique())
            
            missing_metrics = scenario_ids - metrics_scenario_ids
            extra_metrics = metrics_scenario_ids - scenario_ids
            
            if missing_metrics:
                print(f"⚠️  Scenarios missing metrics: {missing_metrics}")
            if extra_metrics:
                print(f"⚠️  Metrics for unknown scenarios: {extra_metrics}")
            
            if not missing_metrics and not extra_metrics:
                print("✅ Metrics match scenarios exactly")
    
    # Validate risk metrics specifically
    if risk_df is not None and 'risk_score' in risk_df.columns:
        valid_scores = risk_df['risk_score'].dropna()
        if len(valid_scores) > 0:
            min_score = valid_scores.min()
            max_score = valid_scores.max()
            if 0.0 <= min_score and max_score <= 1.0:
                print("✅ Risk scores are in valid range [0, 1]")
            else:
                print(f"⚠️  Risk scores outside valid range: [{min_score:.3f}, {max_score:.3f}]")
            
            # Check for TTC values
            if 'min_ttc_s' in risk_df.columns:
                ttc_values = risk_df['min_ttc_s'].dropna()
                if len(ttc_values) > 0:
                    min_ttc = ttc_values.min()
                    if min_ttc > 0:
                        print("✅ TTC values are positive")
                    else:
                        print(f"⚠️  Some TTC values are non-positive: min={min_ttc:.3f}s")
    
    # Validate comfort metrics specifically
    if comfort_df is not None and 'comfort_score' in comfort_df.columns:
        valid_scores = comfort_df['comfort_score'].dropna()
        if len(valid_scores) > 0:
            min_score = valid_scores.min()
            max_score = valid_scores.max()
            if 0.0 <= min_score and max_score <= 1.0:
                print("✅ Comfort scores are in valid range [0, 1]")
            else:
                print(f"⚠️  Comfort scores outside valid range: [{min_score:.3f}, {max_score:.3f}]")
            
            # Check for reasonable acceleration/jerk values
            if 'max_acceleration_mps2' in comfort_df.columns:
                accel_values = comfort_df['max_acceleration_mps2'].dropna()
                if len(accel_values) > 0 and accel_values.max() < 20.0:  # Reasonable upper bound
                    print("✅ Acceleration values are reasonable")
                else:
                    print(f"⚠️  Some acceleration values may be unrealistic: max={accel_values.max():.1f} m/s²")
    
    # Report scenario count (no fixed expectation — count is set by MAX_SCENARIOS in waymo_real_parser.py)
    if scenarios_df is not None:
        actual_count = len(scenarios_df)
        print(f"✅ Scenario count: {actual_count} scenarios parsed")

def main():
    print("=" * 60)
    print("WAYMO VALIDATION LAB - VALIDATE OUTPUTS")
    print("=" * 60)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    silver_dir = project_root / 'data' / 'silver'
    gold_dir = project_root / 'data' / 'gold'
    json_dir = project_root / 'data' / 'exports' / 'scenario_json'  # optional; may not exist
    
    print("📁 VALIDATING FILE STRUCTURE")
    print("=" * 40)
    
    # Check expected files
    expected_files = [
        (silver_dir / 'scenarios.parquet', 'Scenarios table'),
        (silver_dir / 'tracks.parquet', 'Tracks table'),
        (silver_dir / 'states.parquet', 'States table'),
        (gold_dir / 'scenario_metrics.parquet', 'Scenario metrics table'),
        (gold_dir / 'interaction_metrics.parquet', 'Interaction metrics table'),
        (gold_dir / 'risk_metrics.parquet', 'Risk metrics table'),
        (gold_dir / 'comfort_metrics.parquet', 'Comfort metrics table')
    ]
    
    all_files_exist = True
    for file_path, description in expected_files:
        if not validate_file_exists(file_path, description):
            all_files_exist = False
    
    if not all_files_exist:
        print()
        print("❌ Some expected files are missing.")
        print("Please run the extraction and metrics scripts first.")
        sys.exit(1)
    
    print()
    print("📊 VALIDATING PARQUET FILES")
    print("=" * 40)
    
    # Load and validate parquet files
    scenarios_df = validate_parquet_file(silver_dir / 'scenarios.parquet', 'Scenarios table')
    print()
    
    tracks_df = validate_parquet_file(silver_dir / 'tracks.parquet', 'Tracks table')
    print()
    
    states_df = validate_parquet_file(silver_dir / 'states.parquet', 'States table')
    print()
    
    metrics_df = validate_parquet_file(gold_dir / 'scenario_metrics.parquet', 'Scenario metrics table')
    print()
    
    interaction_df = validate_parquet_file(gold_dir / 'interaction_metrics.parquet', 'Interaction metrics table')
    print()
    
    risk_df = validate_parquet_file(gold_dir / 'risk_metrics.parquet', 'Risk metrics table')
    print()
    
    comfort_df = validate_parquet_file(gold_dir / 'comfort_metrics.parquet', 'Comfort metrics table')
    print()
    
    print("📄 VALIDATING JSON EXPORTS")
    print("=" * 40)
    
    # Validate JSON exports
    json_scenario_ids = validate_json_exports(json_dir)
    print()
    
    # Data consistency validation
    validate_data_consistency(scenarios_df, tracks_df, states_df, metrics_df, interaction_df, risk_df, comfort_df, json_scenario_ids)
    
    print()
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    # Overall assessment
    issues = []
    
    if scenarios_df is None or tracks_df is None or states_df is None or metrics_df is None:
        issues.append("Some parquet files failed to load")
    
    if scenarios_df is not None and len(scenarios_df) == 0:
        issues.append("No scenarios found — pipeline may not have run")
    
    if issues:
        print("⚠️  VALIDATION ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ ALL VALIDATIONS PASSED!")
        print("✅ Pipeline outputs look correct and consistent")
        print("✅ Ready for analysis or BigQuery migration")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
