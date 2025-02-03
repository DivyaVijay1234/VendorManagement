import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('automobile_sentiment_analysis.csv')

# Initialize the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Sentiment prediction function using VADER
def predict_sentiment_vader(review):
    sentiment_score = analyzer.polarity_scores(review)
    
    # Determine sentiment based on compound score
    if sentiment_score['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis to the dataset
df['Predicted Sentiment'] = df['Review Text'].apply(predict_sentiment_vader)

# Standardize the 'Sentiment' column and 'Predicted Sentiment' to a consistent format
df['Sentiment'] = df['Sentiment'].str.title()  # Standardize column case for comparison
df['Predicted Sentiment'] = df['Predicted Sentiment'].str.title()  # Standardize column case for comparison

# Calculate percentage of positive/negative/neutral reviews per supplier
supplier_sentiment = df.groupby('Supplier')['Predicted Sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100
supplier_sentiment = supplier_sentiment.rename(columns=str.title)  # Capitalize column names for display

# Display percentages of positive, negative, and neutral reviews for each supplier
print("\nPercentage of Positive, Negative, and Neutral Reviews for Each Supplier:")
print(supplier_sentiment)


