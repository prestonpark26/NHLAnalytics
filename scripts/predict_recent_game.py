import requests
import json
import pandas as pd
import joblib
import sys
import os

# Ensure src is in paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.collectors.nhl_api import get_game_data
from src.data.processors.data_cleaning import filter_shot_events

from src.data.processors.data_cleaning import (
    flatten_nested_columns, remove_unnecessary_columns,
    standardize_coordinates, handle_missing_data, create_goal_target
)
from src.features.feature_engineering import (
    calculate_shot_distance, calculate_shot_angle, extract_game_situation, time_to_seconds
)

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
    
    if 'scoringPlayerId' in X.columns and 'shootingPlayerId' in X.columns:
        X['shootingPlayerId'] = X['shootingPlayerId'].fillna(X['scoringPlayerId'])

    bios_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "player_bios.parquet")
    if os.path.exists(bios_path):
        df_bios = pd.read_parquet(bios_path)
    else:
        df_bios = pd.DataFrame(columns=['playerId', 'position', 'shootsCatches'])
    
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

def main():
    print("="*60)
    print("UTAH MAMMOTH RECENT GAME PIPELINE INFERENCE")
    print("="*60)
    
    # 1. Fetch Utah schedule to find their latest completed game
    print("Fetching Utah Mammoth schedule...")
    try:
        schedule_res = requests.get('https://api-web.nhle.com/v1/club-schedule-season/UTA/now')
        schedule_res.raise_for_status()
        schedule_data = schedule_res.json()
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return
        
    completed_games = [g for g in schedule_data.get('games', []) 
                       if g.get('gameState') in ['OFF', 'FINAL', '7']]
    
    if not completed_games:
        print("No completed games found for Utah Mammoth this season.")
        return
        
    recent_game = completed_games[-1]
    game_id = str(recent_game['id'])
    home_team = recent_game['homeTeam']['abbrev']
    away_team = recent_game['awayTeam']['abbrev']
    home_score = recent_game['homeTeam']['score']
    away_score = recent_game['awayTeam']['score']
    game_date = recent_game['gameDate']
    
    print(f"\\nMost Recent Game Found: {game_date} - {away_team} ({away_score}) @ {home_team} ({home_score}) [Game ID: {game_id}]")
    
    # 2. Fetch play-by-play data using our nhl_api
    print("\\nFetching play-by-play data via NHL API...")
    plays = get_game_data(game_id)
    if not plays:
        print("Could not retrieve play-by-play data.")
        return
        
    df_raw = pd.DataFrame(plays)
    print(f"Retrieved {len(df_raw)} total play events.")
    
    # 3. Filter for shots
    shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_shots = filter_shot_events(df_raw, shot_events)
    
    print(f"Filtered down to {len(df_shots)} shot events.")
    
    # 4. Load pipeline
    model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "nhl_xg_pipeline.pkl")
    print(f"\\nLoading prediction pipeline from {model_path}...")
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        print("Did you run the modeling script/notebook to save the pipeline first?")
        return
        
    # 5. Predict xG
    print("Running predictions through pipeline...")
    # NOTE: Our pipeline handles the raw df directly but requires target columns to be absent.
    # df_shots doesn't have 'is_goal' yet since we just fetched it, so it's perfect.
    try:
        xG_preds = pipeline.predict_proba(df_shots)[:, 1]
    except Exception as e:
        print(f"Prediction error: {e}")
        return

    df_shots['expected_goals'] = xG_preds
    df_shots['is_goal_actual'] = (df_shots['typeDescKey'] == 'goal').astype(int)
    
    # 6. Aggregate results by team
    # Wait, the column in raw data detailing the team is usually 'eventOwnerTeamId'.
    # We'll map the team ID to abbr if possible, or just print by ID.
    print("\\n" + "="*40)
    print(f"GAME SUMMARY: {away_team} @ {home_team}")
    print("="*40)
    
    home_id = recent_game['homeTeam']['id']
    away_id = recent_game['awayTeam']['id']
    
    home_shots = df_shots[df_shots['details'].apply(lambda x: x.get('eventOwnerTeamId') == home_id if isinstance(x, dict) else False)]
    away_shots = df_shots[df_shots['details'].apply(lambda x: x.get('eventOwnerTeamId') == away_id if isinstance(x, dict) else False)]
    
    print(f"Actual Score: {away_team} {away_score} - {home_score} {home_team}")
    
    away_xg = away_shots['expected_goals'].sum()
    home_xg = home_shots['expected_goals'].sum()
    
    print(f"Expected Goals (xG): {away_team} {away_xg:.2f} - {home_xg:.2f} {home_team}")
    
    home_actual_goals = home_shots['is_goal_actual'].sum()
    away_actual_goals = away_shots['is_goal_actual'].sum()
    
    print(f"\\nGoal details (from PxP): {away_team} {away_actual_goals} - {home_actual_goals} {home_team}")
    
    # Identify highest quality chances
    top_chances = df_shots.sort_values(by='expected_goals', ascending=False).head(3)
    print("\\nTop 3 Highest Quality Chances in the Game:")
    for _, chance in top_chances.iterrows():
        details = chance.get('details', {})
        team_id = details.get('eventOwnerTeamId')
        team = home_team if team_id == home_id else (away_team if team_id == away_id else "Unknown")
        player_id = details.get('shootingPlayerId', 'Unknown')
        period = chance.get('periodDescriptor', {}).get('number', '?')
        time = chance.get('timeInPeriod', '?')
        desc = chance.get('typeDescKey', '?')
        xg = chance['expected_goals']
        print(f"  - {team} {desc.upper()} in P{period} @ {time} | xG: {xg:.3f}")

if __name__ == "__main__":
    main()
