import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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