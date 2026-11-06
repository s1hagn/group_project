import pandas as pd

df = pd.read_csv("viral_videos.csv")

# Check values
df.shape
print(df.shape)
df.head()
print(df.head())
df.info()

# Check missing values and duplicated values
df.isnull().sum()
print(df.isnull().sum())
df.duplicated().sum()
print(df.duplicated().sum())

# Remove duplicate videos
df = df.drop_duplicates(subset="video_id")

# Remove missing target values
df = df.dropna(subset=["viral"])

# Validate target and other features
df = df[df["viral"].isin([0, 1])
        & df["upload_hour"].between(0, 23)
        & df["thumbnail_face"].isin([0, 1])
    ].copy()

# Check numerical Variables Summary
df.describe()
print(df.describe())

# Check misspelling
print(df['category'].value_counts()) 
print(df['upload_day'].value_counts())

# Save the cleaned dataset
df.to_csv("cleaned_viral_videos.csv", index=False)
print("Cleaned dataset saved successfully!")