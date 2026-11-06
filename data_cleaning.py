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

# Check numerical Variables Summary
df.describe()
print(df.describe())

# Check categorical variables
print(df['category'].value_counts()) 
print(df['upload_day'].value_counts())

# Save the cleaned dataset
df.to_csv("cleaned_viral_videos.csv", index=False)
print("Cleaned dataset saved successfully!")