from src.data.spotify_client import get_spotify_public_client

sp = get_spotify_public_client()

# prueba con una canción conocida
track_id = "3n3Ppam7vgaVa1iaRUc9Lp"  # Mr. Brightside

features = sp.audio_features([track_id])

print(features)