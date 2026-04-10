"""
Feature Engineering Module

Functions for creating features from cleaned NHL shot data.
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_shot_distance(df: pd.DataFrame) -> pd.Series:
    """
    Calculate distance from shot location to goal.
    
    Args:
        df: DataFrame with xCoord and yCoord columns
        
    Returns:
        Series with distance values
    """
    distance = np.sqrt(
        (89 - df['xCoord'])**2 +
        (0 - df['yCoord'])**2
    )
    return distance


def calculate_shot_angle(df: pd.DataFrame) -> pd.Series:
    """
    Calculate shot angle in degrees from goal line.
    
    Args:
        df: DataFrame with xCoord and yCoord columns
        
    Returns:
        Series with angle values in degrees
    """
    angle = np.degrees(
        np.arctan2(
            df['yCoord'],
            89 - df['xCoord']
        )
    )
    return angle


def extract_game_situation(situation_code: float) -> str:
    """
    Extract game situation from situation code.
    
    Args:
        situation_code: Numeric situation code
        
    Returns:
        String: 'even_strength', 'man_advantage', or 'empty_net'
    """
    if pd.isna(situation_code):
        return 'even_strength'
    
    code_str = str(int(situation_code))
    
    # Check for empty net first (has 0 in first two digits)
    if code_str[0] == '0' or code_str[1] == '0':
        return 'empty_net'
    
    # Common even strength codes
    even_strength_codes = ['1551', '1451', '1541', '1441', '1331', '1351', '1560']
    if code_str in even_strength_codes:
        return 'even_strength'
    
    # Everything else is man-advantage
    return 'man_advantage'


def time_to_seconds(time_str: str) -> Optional[int]:
    """
    Convert time string (MM:SS) to seconds.
    
    Args:
        time_str: Time string in format "MM:SS"
        
    Returns:
        Integer seconds, or None if invalid
    """
    if pd.isna(time_str):
        return None
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return None
    except:
        return None

def add_player_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts shooter and goalie IDs and merges cached player bio features.
    
    Args:
        X: DataFrame from preceding cleaning steps
        
    Returns:
        DataFrame containing attached player demographics and interaction combos
    """
    import os
    X = X.copy()

    # Values are already flattened by clean_data step
    if 'scoringPlayerId' in X.columns and 'shootingPlayerId' in X.columns:
        X['shootingPlayerId'] = X['shootingPlayerId'].fillna(X['scoringPlayerId'])

    bios_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "raw", "player_bios.parquet")
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

