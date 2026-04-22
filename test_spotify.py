from src.data.spotify_client import get_spotify_client

sp = get_spotify_client()

results = sp.current_user_saved_tracks(limit=5)

for item in results['items']:
    print(item['track']['name'])