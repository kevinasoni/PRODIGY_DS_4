import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv("twitter_training.csv", header=None)
val_df = pd.read_csv("twitter_validation.csv", header=None)

# Assign column names
train_df.columns = ["id", "entity", "sentiment", "text"]
val_df.columns = ["id", "entity", "sentiment", "text"]

# Combine datasets
df = pd.concat([train_df, val_df], ignore_index=True)

# Overall sentiment distribution
sentiment_counts = df["sentiment"].value_counts()

plt.figure()
sentiment_counts.plot(kind="bar")
plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Top 5 entities
top_entities = df["entity"].value_counts().head(5).index
entity_df = df[df["entity"].isin(top_entities)]

# Sentiment by entity
entity_sentiment = entity_df.groupby(
    ["entity", "sentiment"]
).size().unstack(fill_value=0)

plt.figure()
entity_sentiment.plot(kind="bar")
plt.title("Sentiment by Top 5 Entities")
plt.xlabel("Entity")
plt.ylabel("Count")
plt.show()
