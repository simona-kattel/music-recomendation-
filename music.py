import pandas as pd

df = pd.read_csv("spotify_merged.csv" )
genre_df = pd.read_csv("genre_music.csv" )

print(df.head())
print(genre_df.head())