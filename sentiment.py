import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('automobile_sentiment_analysis.csv')

# Initialize the Hugging Face sentiment analysis pipeline with a pre-trained BERT model
sentiment_pipeline = pipeline("sentiment-analysis")

# Sentiment prediction function using BERT
def predict_sentiment_bert(review):
    # Use the BERT model to predict sentiment
    sentiment = sentiment_pipeline(review)[0]
    
    # Return the sentiment label (positive, negative, or neutral)
    return sentiment['label']

# Apply sentiment analysis to the dataset using BERT
df['Predicted Sentiment'] = df['Review Text'].apply(predict_sentiment_bert)

# Standardize the 'Sentiment' column and 'Predicted Sentiment' to a consistent format
df['Sentiment'] = df['Sentiment'].str.title()  # Standardize column case for comparison
df['Predicted Sentiment'] = df['Predicted Sentiment'].str.title()  # Standardize column case for comparison

# Calculate percentage of positive/negative/neutral reviews per supplier
supplier_sentiment = df.groupby('Supplier')['Predicted Sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100
supplier_sentiment = supplier_sentiment.rename(columns=str.title)  # Capitalize column names for display

# Display percentages of positive, negative, and neutral reviews for each supplier
print("\nPercentage of Positive, Negative, and Neutral Reviews for Each Supplier:")
print(supplier_sentiment)