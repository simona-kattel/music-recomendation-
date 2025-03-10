import pandas as pd

df = pd.read_csv("spotify_merged.csv" )
genre_df = pd.read_csv("genre_music.csv" )

print(df.head())
print(genre_df.head())


x= dataset.iloc[].values
y= dataset.iloc[].values

from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_split = train_test_split(x,y,test_size=0.2,random_state=42)