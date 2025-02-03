import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.style import apply_common_style

# Configure page settings (must be the first Streamlit command)
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="wide")

# Apply common styling
st.markdown(apply_common_style(), unsafe_allow_html=True)

# Wrap the title in the header div
st.markdown('<div class="header"><h1>Sentiment Analysis Dashboard</h1></div>', unsafe_allow_html=True)

def main():
    # Main content section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
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
            st.subheader("Percentage of Positive, Negative, and Neutral Reviews for Each Supplier")
            st.dataframe(supplier_sentiment)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()