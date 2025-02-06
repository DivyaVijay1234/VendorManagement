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

def load_and_validate_data(df):
    """Load and validate data with flexible column requirements."""
    minimum_required_cols = ['invoice_line_text', 'review_text']
    missing_cols = [col for col in minimum_required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(translate_text(f"Missing essential columns: {missing_cols}", selected_lang_code))
        return None
        
    try:
        # Use invoice_line_text as supplier_name if supplier_name column doesn't exist
        data = df.copy()
        if 'supplier_name' not in data.columns:
            data['supplier_name'] = data['invoice_line_text']
        
        # Clean and prepare data
        data = data[pd.notnull(data.review_text)].reset_index(drop=True)
        return data
    except Exception as e:
        st.error(translate_text(f"Error processing data: {str(e)}", selected_lang_code))
        return None

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
    
    # Model selection (VADER as default)
    model_choice = st.radio(
        translate_text("Choose Sentiment Analysis Model:", selected_lang_code),
        [translate_text("VADER (Faster, Rule-based)", selected_lang_code), 
         translate_text("BERT (More accurate, Deep Learning)", selected_lang_code)],
        index=0  # Set VADER as default
    )
    
    uploaded_file = st.file_uploader(translate_text("Upload CSV file", selected_lang_code), type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            data = load_and_validate_data(df)
            
            if data is None:
                return

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
                
                message = translate_text('Analyzing sentiments using VADER...', selected_lang_code)
                
            else:  # BERT
                sentiment_pipeline = load_bert_model()
                
                def predict_sentiment(review):
                    try:
                        sentiment = sentiment_pipeline(review)[0]
                        return sentiment['label']
                    except Exception:
                        # Fallback to VADER if BERT fails
                        analyzer = load_vader_model()
                        sentiment_score = analyzer.polarity_scores(review)
                        if sentiment_score['compound'] >= 0.05:
                            return 'Positive'
                        elif sentiment_score['compound'] <= -0.05:
                            return 'Negative'
                        else:
                            return 'Neutral'
                
                message = translate_text('Analyzing sentiments using BERT model (this may take longer)...', selected_lang_code)
            
            with st.spinner(message):
                # Apply sentiment analysis to the dataset
                data['Sentiment'] = data['review_text'].apply(predict_sentiment)

                # Calculate percentages by supplier
                supplier_sentiment = data.groupby('supplier_name')['Sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100

                # Display results
                st.subheader(translate_text("Sentiment Analysis Results", selected_lang_code))
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                total_reviews = len(data)
                positive_pct = (data['Sentiment'] == 'Positive').mean() * 100
                negative_pct = (data['Sentiment'] == 'Negative').mean() * 100
                
                col1.metric(
                    translate_text("Total Reviews", selected_lang_code),
                    f"{total_reviews:,}"
                )
                col2.metric(
                    translate_text("Positive Reviews", selected_lang_code),
                    f"{positive_pct:.1f}%"
                )
                col3.metric(
                    translate_text("Negative Reviews", selected_lang_code),
                    f"{negative_pct:.1f}%"
                )

                # Display detailed results
                st.subheader(translate_text("Sentiment Distribution by Supplier", selected_lang_code))
                
                # Visualization
                import plotly.express as px
                fig = px.bar(supplier_sentiment, 
                           title=translate_text(f"Sentiment Distribution Across Suppliers (using {model_choice.split()[0]})", selected_lang_code),
                           labels={'value': translate_text('Percentage', selected_lang_code), 
                                 'supplier_name': translate_text('Supplier Name', selected_lang_code)},
                           barmode='group')
                st.plotly_chart(fig, use_container_width=True)

                # Display raw data
                with st.expander(translate_text("View Detailed Results", selected_lang_code)):
                    st.dataframe(data[['supplier_name', 'review_text', 'Sentiment']])

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()