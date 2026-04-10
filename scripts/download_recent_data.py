import os
import pandas as pd
import json
import glob
import sys
import time
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.collectors.nhl_api import generate_game_ids

def get_game_data_fast(game_id, session, base_url="https://api-web.nhle.com/v1/gamecenter/"):
    url = f"{base_url}{game_id}/play-by-play"
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        time.sleep(0.02) # Fast sleep
        plays = response.json().get("plays", [])
        return plays
    except:
        return []

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old_file = os.path.join(base_dir, "data", "raw", "nhl_raw_plays_2019_2023.parquet")
    new_file = os.path.join(base_dir, "data", "raw", "nhl_raw_plays.parquet")
    
    print(f"Loading existing data from {old_file}...")
    if os.path.exists(old_file):
        df_old = pd.read_parquet(old_file)
        print(f"Found {len(df_old)} existing plays.")
    else:
        df_old = pd.DataFrame()
        print("WARNING: Could not find existing plays. Starting fresh.")
        
    seasons_to_fetch = ["2024", "2025"]
    new_plays = []
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    for season in seasons_to_fetch:
        game_ids = generate_game_ids(season=season, game_type="02", num_games=1312)
        print(f"Fetching season {season}...")
        
        consecutive_empty = 0
        for g_id in game_ids:
            if g_id.endswith("001") or g_id.endswith("500") or g_id.endswith("999"):
                print(f"Fetching {g_id}...")
            plays = get_game_data_fast(g_id, session)
            if plays:
                new_plays.extend(plays)
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                if consecutive_empty > 30:
                    print(f"\\nHit {consecutive_empty} consecutive empty games, assuming end of played season.")
                    break
        
    if new_plays:
        print(f"\\nFetched {len(new_plays)} new plays.")
        df_new = pd.DataFrame(new_plays)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        print(f"Combined total: {len(df_combined)} plays.")
        
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        df_combined.to_parquet(new_file, index=False)
        print(f"Saved complete dataset to {new_file}")
        
        print("Updating notebook and script references...")
        workspace = os.path.join(base_dir, "notebooks")
        for folder in os.listdir(workspace):
            folder_path = os.path.join(workspace, folder)
            if os.path.isdir(folder_path):
                for nb_file in glob.glob(os.path.join(folder_path, '*.ipynb')):
                    with open(nb_file, 'r', encoding='utf-8') as f:
                        data = f.read()
                    if 'nhl_raw_plays_2019_2023.parquet' in data:
                        data = data.replace('nhl_raw_plays_2019_2023.parquet', 'nhl_raw_plays.parquet')
                        with open(nb_file, 'w', encoding='utf-8') as f:
                            f.write(data)
                        print(f"Updated {nb_file}")
                        
        print("Done updating code references!")
    else:
        print("No new plays found.")

if __name__ == "__main__":
    main()
