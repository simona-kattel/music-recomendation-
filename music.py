import pandas as pd

df = pd.read_csv("spotify_merged.csv" )
genre_df = pd.read_csv("genre_music.csv" )

print(df.info()) #checks the missing value of row and coloumn
print(df.head())

print(genre_df.info())
print(genre_df.head())

print(df.duplicated().sum())   # checks if there is any duplicate value
print(genre_df.duplicated().sum())     #here sum is used as we want total count of whatever duplicate and null value is present in the data

print(df.isnull().sum())
print(genre_df.isnull().sum())    #checks if there is any null values

df.drop_duplicates(inplace= True)
genre_df.drop_duplicates(inplace=True)  #drops the duplicate value from the data