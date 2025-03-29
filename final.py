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

# Since the Spotify dataset doesn't have a genre column,
# we merge genre_df with df using the 'track' column.
if 'track' in df.columns and 'track' in genre_df.columns:
    df = df.merge(genre_df[['track', 'genre']], on='track', how='left')
else:
    print("The 'track' column is missing in one of the datasets.")

# Fill missing genres with a placeholder
df['genre'] = df['genre'].fillna("unknown")

# Clean and standardize the genre data: remove extra spaces and convert to lowercase
df['genre'] = df['genre'].str.strip().str.lower()

# Debug: Print unique genres in the merged dataset
print("Unique genres in merged Spotify dataset:", df['genre'].unique())

# Remove duplicates from both datasets
df.drop_duplicates(inplace=True)
genre_df.drop_duplicates(inplace=True)

# Drop rows with more than 40% missing values
df.dropna(thresh=int(df.shape[1] * 0.6), inplace=True)
genre_df.dropna(thresh=int(genre_df.shape[1] * 0.6), inplace=True)

# Standardize column formats for 'track' and 'artist'
df["track"] = df["track"].str.strip()
df["artist"] = df["artist"].str.strip().str.capitalize()

# Handle missing values in numeric columns by filling with median
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Normalize numeric data for better KNN performance
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Encode categorical data (convert genre into numerical values)
if "genre" in df.columns:
    le = LabelEncoder()
    df["genre_encoded"] = le.fit_transform(df["genre"])
else:
    print("The 'genre' column is missing from the dataset!")

# Verify if 'genre_encoded' exists before selecting features
if "genre_encoded" not in df.columns:
    print("The 'genre_encoded' column is missing. Cannot proceed with feature selection!")
else:
    # Selecting features for KNN
    features = ["tempo", "energy", "danceability", "valence", "acousticness", "speechiness", "genre_encoded"]

    # Drop rows with missing values in selected features and reset index
    df.dropna(subset=features, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Scale numerical data for KNN
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])

    # Train KNN model
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')  # 6 because the first one is the song itself
    knn.fit(df_scaled)

def recommend_songs(fav_song, fav_genre):
    """
    Recommend 5 songs based on user's favorite song and genre.
    """
    # Debug: Print the unique genres available in the dataset
    print("Unique genres in dataset:", df['genre'].unique())
    
    # Filter dataset by genre (case-insensitive)
    genre_filtered_df = df[df["genre"] == fav_genre.lower()]
    
    if genre_filtered_df.empty:
        print("\nNo songs found for this genre. Try another genre!")
        # Return an empty DataFrame with appropriate columns
        return pd.DataFrame(columns=["track", "artist", "genre"])

    # Find the selected song (case-insensitive matching)
    song_row = genre_filtered_df[genre_filtered_df["track"].str.lower() == fav_song.lower()]

    if song_row.empty:
        print("\nSorry, your favorite song is not in our dataset.")
        # Return an empty DataFrame with appropriate columns
        return pd.DataFrame(columns=["track", "artist", "genre"])

    # Extract the song index
    song_index = song_row.index[0]

    # Find nearest neighbors based on the selected song's features
    distances, indices = knn.kneighbors([df_scaled[song_index]])
    
    # Get recommended song names (excluding the first one, which is the song itself)
    recommended_songs = df.iloc[indices[0][1:]]
    return recommended_songs[["track", "artist", "genre"]]

# User input for favorite song and genre
fav_song = input("Tell me your favorite song: ").strip()
fav_genre = input("Tell me your favorite genre: ").strip()

# Get recommendations
recommendations = recommend_songs(fav_song, fav_genre)

if not recommendations.empty:
    print("\nRecommended Songs for You:")
    print(recommendations.to_string(index=False))
else:
    print("\nNo recommendations found.")
