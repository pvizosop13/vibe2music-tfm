def enrich_song_text(name, artist):
    text = f"{name} by {artist}"
    
    # reglas simples (puedes ampliarlas)
    keywords = {
        "love": "romantic emotional",
        "night": "chill calm",
        "dance": "party upbeat energetic",
        "remix": "electronic dance",
        "sad": "melancholic slow",
        "summer": "happy upbeat",
        "fire": "intense energetic",
        "dream": "soft chill"
    }
    
    enriched = text.lower()
    
    for key, value in keywords.items():
        if key in enriched:
            enriched += " " + value
    
    return enriched