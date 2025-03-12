import pandas as pd

# Load data
df = pd.read_csv("spotify_merged.csv")
genre_df = pd.read_csv("genre_music.csv")

# Check missing values and duplicates
print(df.info())
print(df.head())

print(genre_df.info())
print(genre_df.head())

print(f"Duplicates in df: {df.duplicated().sum()}")
print(f"Duplicates in genre_df: {genre_df.duplicated().sum()}")

print("Missing values in df:\n", df.isnull().sum())
print("Missing values in genre_df:\n", genre_df.isnull().sum())

# Handle missing values
df.dropna(thresh=int(df.shape[1] * 0.6), inplace=True)  # Drop rows with >40% missing values
genre_df.dropna(thresh=int(genre_df.shape[1] * 0.6), inplace=True)

# Fill missing values
df["popularity"].fillna(df["popularity"].median(), inplace=True)  # Fill popularity with median
if "decade" in df.columns:
    df["decade"].fillna("unknown", inplace=True)  # Fill missing decade values

# Drop duplicate rows
df.drop_duplicates(inplace=True)
genre_df.drop_duplicates(inplace=True)

# Fix column formatting
df["track"] = df["track"].str.strip()
df["artist"] = df["artist"].str.strip().str.capitalize()

if "genre" in df.columns:
    df["genre"] = df["genre"].str.lower()
if "genre" in genre_df.columns:
    genre_df["genre"] = genre_df["genre"].str.lower()

# Save cleaned data
df.to_csv("spotify_cleaned.csv", index=False)
genre_df.to_csv("genre_music_cleaned.csv", index=False)

print(df.info())
print(df.head())
