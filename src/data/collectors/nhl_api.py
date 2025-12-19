"""
NHL API Data Collection Module

Functions for fetching play-by-play data from the NHL API.
"""

import requests
import time
from typing import List, Optional, Dict, Any


def get_game_data(game_id: str, base_url: str = "https://api-web.nhle.com/v1/gamecenter/") -> Optional[List[Dict[str, Any]]]:
    """
    Fetches play-by-play data for a given game ID.
    
    Args:
        game_id: Game ID in format YYYYGGGGGG (e.g., "2023020001")
        base_url: Base URL for NHL API
        
    Returns:
        List of play dictionaries, or None if error occurred
    """
    url = f"{base_url}{game_id}/play-by-play"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Add small delay to be respectful to the API
        time.sleep(0.1)
        
        data = response.json()
        plays = data.get("plays", [])
        return plays
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for game {game_id}: {e}")
        return None


def generate_game_ids(season: str = "2023", game_type: str = "02", num_games: int = 1312) -> List[str]:
    """
    Generate list of game IDs for a season.
    
    Args:
        season: Season year (e.g., "2023")
        game_type: Game type code ("01" = Preseason, "02" = Regular Season, "03" = Playoffs)
        num_games: Number of games in the season
        
    Returns:
        List of game ID strings
    """
    return [f"{season}{game_type}{str(i).zfill(4)}" for i in range(1, num_games + 1)]

