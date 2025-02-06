import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from utils.style import apply_common_style
from utils.translation import translate_text, get_language_code


# Configure page settings (must be the first Streamlit command)
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="wide")

# Apply common styling
st.markdown(apply_common_style(), unsafe_allow_html=True)
# Sidebar for language selection
languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Urdu', 'Gujarati', 'Punjabi', 'Malayalam', 'Odia', 'Kannada', 'Assamese', 'Maithili', 'Sanskrit']
selected_language = st.sidebar.selectbox("Select Language", languages)
selected_lang_code = get_language_code(selected_language)
# Wrap the title in the header div
st.markdown(f'<div class="header"><h1>{translate_text("Sentiment Analysis Dashboard", selected_lang_code)}</h1></div>', unsafe_allow_html=True)

def main():
    # Initialize the models with caching
    @st.cache_resource
    def load_bert_model():
        return pipeline("sentiment-analysis")
    
    @st.cache_resource
    def load_vader_model():
        return SentimentIntensityAnalyzer()

    # Main content section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    # Model selection
    model_choice = st.radio(
        translate_text("Choose Sentiment Analysis Model:", selected_lang_code),
        [translate_text("VADER (Faster, Rule-based)", selected_lang_code), 
         translate_text("BERT (More accurate, Deep Learning)", selected_lang_code)]
    )
    
    uploaded_file = st.file_uploader(translate_text("Upload CSV file", selected_lang_code), type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if "VADER" in model_choice:
                analyzer = load_vader_model()
                
                def predict_sentiment(review):
                    sentiment_score = analyzer.polarity_scores(review)
                    if sentiment_score['compound'] >= 0.05:
                        return 'Positive'
                    elif sentiment_score['compound'] <= -0.05:
                        return 'Negative'
                    else:
                        return 'Neutral'
                
                message = 'Analyzing sentiments using VADER...'
                
            else:  # BERT
                sentiment_pipeline = load_bert_model()
                
                def predict_sentiment(review):
                    sentiment = sentiment_pipeline(review)[0]
                    return sentiment['label']
                
                message = 'Analyzing sentiments using BERT model (this may take longer)...'
            
            with st.spinner(message):
                # Apply sentiment analysis to the dataset
                df['Predicted Sentiment'] = df['Review Text'].apply(predict_sentiment)

                # Standardize the format
                df['Sentiment'] = df['Sentiment'].str.title()
                df['Predicted Sentiment'] = df['Predicted Sentiment'].str.title()

                # Calculate percentages
                supplier_sentiment = df.groupby('Supplier')['Predicted Sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100
                supplier_sentiment = supplier_sentiment.rename(columns=str.title)

                # Display results
                st.subheader("Favourability of Reviews for Each Supplier")
                st.dataframe(supplier_sentiment)

                # Visualization
                st.subheader("Sentiment Distribution by Supplier")
                import plotly.express as px
                fig = px.bar(supplier_sentiment, 
                           title=f"Sentiment Distribution Across Suppliers (using {model_choice.split()[0]})",
                           labels={'value': 'Percentage', 'Supplier': 'Supplier Name'},
                           barmode='group')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()