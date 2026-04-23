def get_all_liked_songs(sp):
    songs = []
    results = sp.current_user_saved_tracks(limit=50)
    
    while results:
        for item in results['items']:
            track = item['track']
            songs.append({
                "id": track["id"],
                "name": track["name"],
                "artist": track["artists"][0]["name"]
            })
        
        if results['next']:
            results = sp.next(results)
        else:
            results = None

    return songs