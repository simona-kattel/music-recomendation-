import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

# Load data
df = pd.read_csv("spotify_merged.csv")
genre_df = pd.read_csv("genre_music.csv")

# Display dataset info and first few rows
print("Original Spotify Data:")
print(df.info())
print(df.head())
print("\nOriginal Genre Data:")
print(genre_df.info())
print(genre_df.head())

# Check for duplicates
df.drop_duplicates(inplace=True)
genre_df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {df.duplicated().sum()} in df, {genre_df.duplicated().sum()} in genre_df")

# Check for missing values and drop rows with more than 40% missing values
df.dropna(thresh=int(df.shape[1] * 0.6), inplace=True)
genre_df.dropna(thresh=int(genre_df.shape[1] * 0.6), inplace=True)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print(genre_df.isnull().sum())

# Standardize column formats
df["track"] = df["track"].str.strip()
df["artist"] = df["artist"].str.strip().str.capitalize()

# Convert genres to lowercase if they exist
if "genre" in df.columns:
    df["genre"] = df["genre"].str.lower()
if "genre" in genre_df.columns:
    genre_df["genre"] = genre_df["genre"].str.lower()

# Identify numeric columns for normalization
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

# Handle missing values in numeric columns (fill with median)
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Normalize numeric data to improve KNN performance
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Display cleaned dataset info
print("\nCleaned Spotify Data:")
print(df.info())
print(df.head())

# Visualize outliers with boxplots
plt.figure(figsize=(6, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(6, 10))
sns.boxplot(data=genre_df)
plt.xticks(rotation=90)
plt.show()

# Save cleaned data
df.to_csv("spotify_cleaned.csv", index=False)
genre_df.to_csv("genre_music_cleaned.csv", index=False)

print("\nData cleaning completed! Ready for KNN model.")

# Load cleaned data
df = pd.read_csv("spotify_cleaned.csv")
genre_df = pd.read_csv("genre_music_cleaned.csv")

# Encode categorical data (convert genre into numerical values)
if "genre" in df.columns:
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre"])

# Selecting features for KNN
features = ["tempo", "energy", "danceability", "valence", "acousticness", "speechiness", "genre_encoded"]

# Handle missing numeric values
df.dropna(subset=features, inplace=True)

# Scale numerical data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Train KNN model
knn = NearestNeighbors(n_neighbors=6, metric='euclidean')  # 6 because first one will be the song itself
knn.fit(df_scaled)

def recommend_songs(fav_song, fav_genre):
    """Recommend 5 songs based on user's favorite song and genre"""
    # Filter dataset by genre
    genre_filtered_df = df[df["genre"] == fav_genre]

    if genre_filtered_df.empty:
        print("No songs found for this genre. Try another genre!")
        return []
    
    # Find the selected song
    song_row = genre_filtered_df[genre_filtered_df["track"].str.lower() == fav_song.lower()]
    
    if song_row.empty:
        print("Sorry, your favorite song is not in our dataset.")
        return []
    
    # Extract the song index
    song_index = song_row.index[0]
    
    # Find nearest neighbors
    distances, indices = knn.kneighbors([df_scaled[song_index]])
    
    # Get recommended song names
    recommended_songs = df.iloc[indices[0][1:]]  # Exclude the first song (itself)
    return recommended_songs[["track", "artist", "genre"]]

# Get user input
fav_song = input("Tell me your favorite song: ")
fav_genre = input("Tell me your favorite genre: ")

# Get recommendations
recommendations = recommend_songs(fav_song, fav_genre)
print("\nRecommended Songs for You:")
print(recommendations)
