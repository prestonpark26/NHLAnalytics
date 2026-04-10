import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import joblib
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.processors.data_cleaning import (
    filter_shot_events, flatten_nested_columns, remove_unnecessary_columns,
    standardize_coordinates, handle_missing_data, create_goal_target
)
from src.features.feature_engineering import (
    calculate_shot_distance, calculate_shot_angle, extract_game_situation, time_to_seconds
)

print("Loading raw data...")
INPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nhl_raw_plays.parquet")
df_raw = pd.read_parquet(INPUT_FILE)

print("Filtering shots and creating target...")
shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
df_shots = filter_shot_events(df_raw, shot_events)
df_shots = create_goal_target(df_shots)

X = df_shots.drop(columns=['is_goal'])
y = df_shots['is_goal']

def clean_data(X):
    X = flatten_nested_columns(X)
    X = remove_unnecessary_columns(X)
    X = standardize_coordinates(X)
    X = handle_missing_data(X)
    return X

def engineer_features(X):
    X = X.copy()
    X['distance'] = calculate_shot_distance(X)
    X['angle'] = calculate_shot_angle(X)
    X['game_situation'] = X['situationCode'].apply(extract_game_situation)
    X['is_even_strength'] = (X['game_situation'] == 'even_strength').astype(int)
    X['is_empty_net'] = (X['game_situation'] == 'empty_net').astype(int)
    X['is_man_advantage'] = (X['game_situation'] == 'man_advantage').astype(int)
    X['time_in_period_seconds'] = X['timeInPeriod'].apply(time_to_seconds)
    X['time_remaining_seconds'] = X['timeRemaining'].apply(time_to_seconds)
    X['time_remaining_game'] = (X['period'] - 1) * 1200 + X['time_remaining_seconds'].fillna(0)
    X['is_regulation'] = (X['periodType'] == 'REG').astype(int)
    X['is_overtime'] = (X['periodType'] == 'OT').astype(int)
    X['is_shootout'] = (X['periodType'] == 'SO').astype(int)
    X['is_late_game'] = ((X['period'] <= 3) & (X['time_remaining_seconds'] <= 300)).astype(int)
    X['is_early_period'] = (X['time_in_period_seconds'] <= 120).astype(int)
    X['distance_angle_interaction'] = X['distance'] * X['angle'].abs()
    X['normalized_distance'] = X['distance'] / 100
    X['normalized_angle'] = X['angle'].abs() / 90
    X['shot_quality_score'] = X['normalized_distance'] * 0.6 + X['normalized_angle'] * 0.4
    X['is_high_danger'] = ((X['distance'] < 25) & (X['angle'].abs() < 30)).astype(int)
    X['is_medium_danger'] = ((X['distance'] < 40) & (X['angle'].abs() < 45) & ~X['is_high_danger']).astype(int)
    X['is_low_danger'] = (~X['is_high_danger'] & ~X['is_medium_danger']).astype(int)
    
    X_sorted = X.sort_values('eventId').reset_index(drop=False)
    X_sorted['prev_event_id'] = X_sorted['eventId'].shift(1)
    X_sorted['event_id_diff'] = X_sorted['eventId'] - X_sorted['prev_event_id']
    X_sorted['is_potential_rebound'] = ((X_sorted['event_id_diff'] <= 5) & (X_sorted['event_id_diff'] > 0)).astype(int)
    X['is_potential_rebound'] = X_sorted.set_index('index')['is_potential_rebound'].fillna(0).astype(int)
    
    common_shot_types = ['wrist', 'snap', 'slap', 'tip-in', 'backhand', 'deflected']
    for shot_type in common_shot_types:
        X[f'is_{shot_type}'] = (X['shotType'] == shot_type).astype(int)
        
    X['is_offensive_zone'] = (X['zoneCode'] == 'O').astype(int)
    X['is_defensive_zone'] = (X['zoneCode'] == 'D').astype(int)
    X['is_neutral_zone'] = (X['zoneCode'] == 'N').astype(int)
    return X


def add_player_features(X):
    X = X.copy()
    
    # Fix Target Leakage: Goals often use scoringPlayerId instead of shootingPlayerId
    if 'scoringPlayerId' in X.columns and 'shootingPlayerId' in X.columns:
        X['shootingPlayerId'] = X['shootingPlayerId'].fillna(X['scoringPlayerId'])

    bios_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "player_bios.parquet")
    if os.path.exists(bios_path):
        df_bios = pd.read_parquet(bios_path)
    else:
        df_bios = pd.DataFrame(columns=['playerId', 'position', 'shootsCatches'])
    
    # Use mapping to avoid merge scrambling
    pos_map = dict(zip(df_bios['playerId'], df_bios['position']))
    hand_map = dict(zip(df_bios['playerId'], df_bios['shootsCatches']))

    X['shooter_position'] = X['shootingPlayerId'].map(pos_map).fillna('Unknown')
    X['shooter_handedness'] = X['shootingPlayerId'].map(hand_map).fillna('Unknown')
    X['goalie_handedness'] = X['goalieInNetId'].map(hand_map).fillna('Unknown')
    
    X['shooter_off_wing'] = (
        ((X['shooter_handedness'] == 'L') & (X['yCoord'] < 0)) |
        ((X['shooter_handedness'] == 'R') & (X['yCoord'] > 0))
    ).astype(int)
    
    cat_columns = ['shootingPlayerId', 'goalieInNetId', 'shooter_position', 
                   'shooter_handedness', 'goalie_handedness']
    for col in cat_columns:
        X[col] = X[col].astype(str).astype('category')
        
    return X


def drop_unnecessary_features(X):
    exclude_cols = [
        'eventId', 'timeInPeriod', 'timeRemaining', 'situationCode', 
        'typeDescKey', 'typeCode', 'sortOrder', 'assist1PlayerId', 
        'assist1PlayerTotal', 'assist2PlayerId', 'assist2PlayerTotal',
        'awaySOG', 'awayScore', 'blockingPlayerId', 'eventOwnerTeamId', 
        'homeSOG', 'homeScore', 'scoringPlayerId', 
        'scoringPlayerTotal', 'xCoord', 'yCoord', 
        'zoneCode', 'period', 'periodType', 'shotType', 'homeTeamDefendingSide',
        'game_situation', 'details', 'periodDescriptor'
    ]
    cols_to_drop = [c for c in exclude_cols if c in X.columns]
    X_clean = X.drop(columns=cols_to_drop)
    
    # Drop unknown object columns, keep categories
    for col in X_clean.columns:
        if X_clean[col].dtype == 'object':
            X_clean = X_clean.drop(columns=[col])
            
    # Fill NA for numerics only
    numeric_cols = X_clean.select_dtypes(exclude=['category']).columns
    X_clean[numeric_cols] = X_clean[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Sort columns alphabetically to prevent feature_name mismatch between training and inference
    X_clean = X_clean.reindex(sorted(X_clean.columns), axis=1)
    
    return X_clean

preprocessor = Pipeline([
    ('data_cleaning', FunctionTransformer(clean_data)),
    ('feature_engineering', FunctionTransformer(engineer_features)),
    ('player_features', FunctionTransformer(add_player_features)),
    ('feature_selection', FunctionTransformer(drop_unnecessary_features))
])

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        enable_categorical=True,
        objective='binary:logistic', eval_metric='auc', n_estimators=100, max_depth=6,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0))
])

print("Training final pipeline on all data...")
full_pipeline.fit(X.reset_index(drop=True), y.reset_index(drop=True))

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "models"), exist_ok=True)
model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "nhl_xg_pipeline.pkl")
joblib.dump(full_pipeline, model_path)
print(f"Saved full prediction pipeline to {model_path}")
