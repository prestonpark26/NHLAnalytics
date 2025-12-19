"""
Data Cleaning Module

Functions for cleaning and preprocessing NHL play-by-play data.
"""

import pandas as pd
import numpy as np
from typing import List


def filter_shot_events(df: pd.DataFrame, shot_events: List[str] = None) -> pd.DataFrame:
    """
    Filter DataFrame to include only shot-related events.
    
    Args:
        df: Raw play-by-play DataFrame
        shot_events: List of event types to include. Default: ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
        
    Returns:
        Filtered DataFrame with only shot events
    """
    if shot_events is None:
        shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    
    df_shots = df[df["typeDescKey"].isin(shot_events)].copy()
    return df_shots


def flatten_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten nested JSON columns (details and periodDescriptor).
    
    Args:
        df: DataFrame with nested columns
        
    Returns:
        DataFrame with flattened columns
    """
    # Flatten the details column
    df_details_flat = pd.json_normalize(df["details"])
    
    # Flatten the period column
    df_period_flat = pd.json_normalize(df["periodDescriptor"])
    
    # Reset indices to ensure proper joining
    df = df.reset_index(drop=True)
    df_details_flat = df_details_flat.reset_index(drop=True)
    df_period_flat = df_period_flat.reset_index(drop=True)
    
    # Join the flattened columns
    df_clean = pd.concat([df, df_details_flat, df_period_flat], axis=1)
    
    # Drop original nested columns
    df_clean.drop(columns=["details", "periodDescriptor"], inplace=True)
    
    # Check for and handle duplicate column names
    if df_clean.columns.duplicated().any():
        # Get duplicate column names
        duplicates = df_clean.columns[df_clean.columns.duplicated()].unique()
        print(f"Warning: Found duplicate columns: {list(duplicates)}")
        
        # Remove duplicate columns (keep first occurrence)
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        print(f"Removed duplicate columns. New shape: {df_clean.shape}")
    
    # Rename 'number' column to 'period' (if it exists and isn't already named 'period')
    if 'number' in df_clean.columns and 'period' not in df_clean.columns:
        df_clean.rename(columns={"number": "period"}, inplace=True)
    elif 'number' in df_clean.columns and 'period' in df_clean.columns:
        # If both exist, drop 'number' and keep 'period'
        df_clean.drop(columns=["number"], inplace=True)
    
    return df_clean


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns not relevant for shot analysis.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame with unnecessary columns removed
    """
    # Remove clip/URL columns
    clip_columns = [
        "highlightClipSharingUrl", "highlightClip", "highlightClipSharingUrlFr",
        "highlightClipFr", "discreteClip", "discreteClipFr", "pptReplayUrl"
    ]
    df = df.drop(columns=[col for col in clip_columns if col in df.columns])
    
    # Remove other unnecessary columns
    columns_to_drop = [
        # Faceoff-related
        'winningPlayerId', 'losingPlayerId',
        # Penalty-related
        'committedByPlayerId', 'drawnByPlayerId', 'servedByPlayerId',
        'reason', 'secondaryReason', 'duration',
        # Hit-Related
        'hittingPlayerId', 'hitteePlayerId',
        # Other
        'playerId',  # Generic column - we use shootingPlayerId instead
        'descKey',  # Redundant because we have typeDescKey
        'maxRegulationPeriods',  # All NHL games have the same number
    ]
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df


def standardize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize shot coordinates so all shots are aimed at the positive-x net.
    
    Args:
        df: DataFrame with xCoord and yCoord columns
        
    Returns:
        DataFrame with standardized coordinates
    """
    df = df.copy()
    
    # Ensure coordinates are numeric
    df["xCoord"] = pd.to_numeric(df["xCoord"])
    df["yCoord"] = pd.to_numeric(df["yCoord"])
    
    # Find shots at negative net (xCoord < 0)
    flip_mask = df["xCoord"] < 0
    
    # Flip coordinates for negative-x shots
    df.loc[flip_mask, "xCoord"] = df.loc[flip_mask, "xCoord"] * -1
    df.loc[flip_mask, "yCoord"] = df.loc[flip_mask, "yCoord"] * -1
    
    return df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Fill missing shotType with 'unknown'
    if 'shotType' in df.columns:
        df["shotType"] = df["shotType"].fillna("unknown")
    
    # Fill missing assist columns with "None" and ensure they're strings
    assist_columns = ['assist1PlayerId', 'assist1PlayerTotal', 'assist2PlayerId', 'assist2PlayerTotal']
    for col in assist_columns:
        if col in df.columns:
            # Convert to string first to handle mixed types, then fill NaN
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', 'None')
            df[col] = df[col].replace('NaN', 'None')
            df[col] = df[col].fillna("None")
    
    return df


def create_goal_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable for goals.
    
    Args:
        df: DataFrame with typeDescKey column
        
    Returns:
        DataFrame with is_goal column added
    """
    df = df.copy()
    df["is_goal"] = np.where(df["typeDescKey"] == "goal", 1, 0)
    return df

