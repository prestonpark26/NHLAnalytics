import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def get_player_bio(player_id, session):
    url = f"https://api-web.nhle.com/v1/player/{int(player_id)}/landing"
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            'playerId': int(player_id),
            'firstName': data.get('firstName', {}).get('default', ''),
            'lastName': data.get('lastName', {}).get('default', ''),
            'position': data.get('position', ''),
            'shootsCatches': data.get('shootsCatches', '')
        }
    except Exception as e:
        return {'playerId': int(player_id), 'position': 'Unknown', 'shootsCatches': 'Unknown'}

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plays_file = os.path.join(base_dir, "data", "raw", "nhl_raw_plays.parquet")
    bios_file = os.path.join(base_dir, "data", "raw", "player_bios.parquet")
    
    print(f"Loading {plays_file} to find unique players...")
    df = pd.read_parquet(plays_file)
    
    # We need to extract from details column
    # details is an array of dicts or records
    # Let's cleanly extract shootingPlayerId and goalieInNetId
    # Assuming df has 'details' column which are dicts
    
    # We can use pd.json_normalize if details is literally a dict, or we can use custom apply
    print("Extracting IDs...")
    def get_id(x, key):
        if isinstance(x, dict):
            return x.get(key)
        return None
        
    shooter_ids = df['details'].apply(lambda x: get_id(x, 'shootingPlayerId')).dropna().unique()
    goalie_ids = df['details'].apply(lambda x: get_id(x, 'goalieInNetId')).dropna().unique()
    
    all_unique_ids = set([int(x) for x in shooter_ids]).union(set([int(x) for x in goalie_ids]))
    print(f"Found {len(all_unique_ids)} unique player IDs.")
    
    existing_bios = []
    if os.path.exists(bios_file):
        df_bios = pd.read_parquet(bios_file)
        existing_bios = df_bios['playerId'].tolist()
        print(f"Found {len(existing_bios)} already cached.")
    else:
        df_bios = pd.DataFrame()
        
    ids_to_fetch = [pid for pid in all_unique_ids if pid not in existing_bios]
    print(f"Need to fetch {len(ids_to_fetch)} new players.")
    
    if not ids_to_fetch:
        print("All players cached!")
        return
        
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    new_records = []
    # Using ThreadPool to speed it up since it's just IO
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {executor.submit(get_player_bio, pid, session): pid for pid in ids_to_fetch}
        
        fetched = 0
        for future in as_completed(future_to_id):
            pid = future_to_id[future]
            try:
                res = future.result()
                new_records.append(res)
                fetched += 1
                if fetched % 200 == 0:
                    print(f"Fetched {fetched}/{len(ids_to_fetch)}...")
            except Exception as exc:
                print(f"Player {pid} generated an exception: {exc}")
                
    df_new = pd.DataFrame(new_records)
    df_final = pd.concat([df_bios, df_new], ignore_index=True) if not df_bios.empty else df_new
    df_final.to_parquet(bios_file, index=False)
    print(f"Saved {len(df_final)} player bios to {bios_file}")

if __name__ == '__main__':
    main()
