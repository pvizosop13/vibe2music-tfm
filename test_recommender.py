from src.model.recommender import load_dataset, recommend_songs

df = load_dataset()

vibe = "motivation for gym"

results = recommend_songs(vibe, df)

print(results)