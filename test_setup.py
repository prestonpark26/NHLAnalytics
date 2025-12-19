"""
Quick test script to verify the setup is working correctly.
Run this from the project root: python test_setup.py
"""

import sys
import os

print("=" * 60)
print("Testing NHL Analytics Setup")
print("=" * 60)

# Test 1: Check Python version
print("\n1. Python version:")
print(f"   {sys.version}")

# Test 2: Check if data file exists
print("\n2. Checking data files:")
data_file = "data/raw/nhl_raw_plays.parquet"
if os.path.exists(data_file):
    print(f"   ✓ Found: {data_file}")
    import pandas as pd
    df = pd.read_parquet(data_file)
    print(f"   ✓ File loads successfully (shape: {df.shape})")
else:
    print(f"   ✗ Missing: {data_file}")
    print("   → Run notebook 01_data_collection.ipynb first")

# Test 3: Check imports
print("\n3. Testing imports:")
try:
    sys.path.append('.')
    from src.data.processors.data_cleaning import (
        filter_shot_events,
        flatten_nested_columns,
        remove_unnecessary_columns,
        standardize_coordinates,
        handle_missing_data,
        create_goal_target
    )
    print("   ✓ All data cleaning functions import successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")

try:
    from src.data.collectors.nhl_api import get_game_data, generate_game_ids
    print("   ✓ NHL API functions import successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")

try:
    from src.features.feature_engineering import (
        calculate_shot_distance,
        calculate_shot_angle,
        extract_game_situation,
        time_to_seconds
    )
    print("   ✓ Feature engineering functions import successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")

# Test 4: Check required packages
print("\n4. Checking required packages:")
required_packages = [
    'pandas', 'numpy', 'requests', 'matplotlib', 
    'seaborn', 'sklearn', 'xgboost', 'jupyter'
]

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - NOT INSTALLED")

print("\n" + "=" * 60)
print("Setup test complete!")
print("=" * 60)

