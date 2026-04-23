from src.model.recommender import load_dataset, recommend_songs

df = load_dataset()

vibe = "chill music for study"

results = recommend_songs(vibe, df)

print(results)