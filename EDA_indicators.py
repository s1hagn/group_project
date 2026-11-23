import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("cleaned_viral_videos.csv")

# Check the dataset
print("Dataset Shape:")
print(df.shape)

# Count videos in each category
print("\nVideo Categories:")
print(df["category"].value_counts())

# Step 2: Calculate viral rate by category
viral_rate = df.groupby("category")["viral"].mean() * 100

# Sort from highest to lowest
viral_rate = viral_rate.sort_values(ascending=False)

print("\nViral Rate by Category (%):")
print(viral_rate.round(2))

# Plot viral rate by category
viral_rate.plot(kind="bar", figsize=(10, 5))

plt.title("Viral Rate by Video Category")
plt.xlabel("Video Category")
plt.ylabel("Viral Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()

# Save chart 1 as PNG
plt.savefig("viral_rate_by_category.png", dpi=300)
print("Chart saved successfully!")


# Step 3: Average video duration by category
avg_duration = df.groupby("category")["duration_sec"].mean()

# Convert seconds to minutes
avg_duration = avg_duration / 60

# Sort from shortest to longest
avg_duration = avg_duration.sort_values()

print("\nAverage Video Duration by Category (minutes):")
print(avg_duration.round(2))

# Step 4: Group videos by duration

df["duration_group"] = pd.cut(df["duration_sec"],bins=[0, 300, 600, float("inf")],labels=["Short", "Medium", "Long"],include_lowest=True)

print("\nVideo Count by Duration Group:")
print(df["duration_group"].value_counts())

print("\nViral Rate by Duration Group (%):")
print((df.groupby("duration_group", observed=True)["viral"].mean() * 100).round(2))

# Chart 2 : Viral Rate by Duration Group

duration_rate = df.groupby("duration_group", observed=True)["viral"].mean() * 100

duration_rate.plot(kind="bar", figsize=(7, 5))

plt.title("Viral Rate by Duration Group")
plt.xlabel("Video Duration")
plt.ylabel("Viral Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("viral_rate_by_duration.png", dpi=300)
plt.close()
print("Duration chart saved successfully!")


# Step 5: Viral rate by category and duration group

category_duration = df.groupby(["category", "duration_group"],observed=True)["viral"].agg(["count", "mean"])

category_duration["viral_rate"] = (category_duration["mean"] * 100)

print("\nViral Rate by Category and Duration:")
print(category_duration[["count", "viral_rate"]].round(2))


# Chart 3: Viral Rate by Category and Duration

# Calculate viral rate for each combination
heatmap_data = df.groupby(["category", "duration_group"],observed=True)["viral"].mean().unstack() * 100

# Sort categories by overall viral rate
category_order = (df.groupby("category")["viral"].mean().sort_values(ascending=False).index)

heatmap_data = heatmap_data.reindex(category_order)


# Create heatmap
plt.figure(figsize=(9, 7))

plt.imshow(heatmap_data, cmap="Blues", aspect="auto")

plt.colorbar(label="Viral Rate (%)")
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
plt.yticks(range(len(heatmap_data.index)),heatmap_data.index)

plt.title("Viral Rate by Category and Duration")
plt.xlabel("Video Duration")
plt.ylabel("Video Category")

plt.tight_layout()
plt.savefig("viral_rate_heatmap.png", dpi=300)
plt.close()

print("Heatmap saved successfully!")

# Step 6: Median subscriber count by category

median_subscribers = df.groupby("category")["subscriber_count"].median()

# Sort from highest to lowest
median_subscribers = median_subscribers.sort_values(ascending=False)

print("\nMedian Subscribers by Category:")
print(median_subscribers.round(0))


# Chart 4: Median Subscribers by Category

median_subscribers.plot(kind="bar",figsize=(10, 5))

plt.title("Median Subscriber Count by Category")
plt.xlabel("Video Category")
plt.ylabel("Median Subscriber Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("median_subscribers_by_category.png", dpi=300)
plt.close()

print("Subscriber chart saved successfully!")
