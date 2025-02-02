import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('automobile_sentiment_analysis.csv')

# Load a pretrained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # A pretrained sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Preprocessing Function: Clean text
def clean_text(text):
    # Remove special characters, URLs, and digits
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    return text

# Tokenization and Sentiment Prediction in Batches
def predict_sentiment_batch(reviews):
    # Clean all reviews first
    cleaned_reviews = [clean_text(review) for review in reviews]
    
    # Tokenize the batch of reviews
    tokens = tokenizer(cleaned_reviews, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**tokens)
        probabilities = softmax(outputs.logits, dim=1)
        sentiments = torch.argmax(probabilities, dim=1).tolist()  # List of 0s and 1s
    
    # Map sentiments to human-readable labels
    return ["Positive" if sentiment == 1 else "Negative" for sentiment in sentiments]

# Apply batch sentiment prediction to the dataset
df['Predicted Sentiment'] = predict_sentiment_batch(df['Review Text'].tolist())

# Calculate percentage of positive/negative reviews per supplier
supplier_sentiment = df.groupby('Supplier')['Predicted Sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100
supplier_sentiment = supplier_sentiment.rename(columns=str.title)  # Capitalize column names for display

# Display percentages of positive and negative reviews for each supplier
print("\nPercentage of Positive and Negative Reviews for Each Supplier:")
print(supplier_sentiment)

# Calculate the overall accuracy of the model
df['Sentiment'] = df['Sentiment'].str.title()  # Standardize column case for comparison
accuracy = accuracy_score(df['Sentiment'], df['Predicted Sentiment'])
print(f"\nOverall Accuracy of the Sentiment Analysis Model: {accuracy * 100:.2f}%")
