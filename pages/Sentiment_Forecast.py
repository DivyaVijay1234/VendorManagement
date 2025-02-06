import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.style import apply_common_style
from utils.translation import translate_text, get_language_code

# Configure page settings
st.set_page_config(page_title="Vendor Demand Forecast", page_icon="📊", layout="wide")

# Apply common styling
st.markdown(apply_common_style(), unsafe_allow_html=True)

# Sidebar for language selection
languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Urdu', 'Gujarati', 'Punjabi', 'Malayalam', 'Odia', 'Kannada', 'Assamese', 'Maithili', 'Sanskrit']
selected_language = st.sidebar.selectbox("Select Language", languages)
selected_lang_code = get_language_code(selected_language)

# Wrap the title in the header div
st.markdown(f'<div class="header"><h1>{translate_text("Vendor Demand Forecast", selected_lang_code)}</h1></div>', unsafe_allow_html=True)

def load_and_validate_data(df):
    """Load and validate data with flexible column requirements."""
    minimum_required_cols = ['job_card_date', 'invoice_line_text', 'supplier_name', 'review_text']
    missing_cols = [col for col in minimum_required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(translate_text(f"Missing essential columns: {missing_cols}", selected_lang_code))
        return None
        
    try:
        data = df[pd.notnull(df.invoice_line_text) & pd.notnull(df.review_text)].reset_index(drop=True)
        
        # Convert job_card_date to datetime with flexible format handling
        try:
            data['job_card_date'] = pd.to_datetime(data['job_card_date'], format='%d-%m-%y')
        except ValueError:
            try:
                data['job_card_date'] = pd.to_datetime(data['job_card_date'], format='%d-%m-%Y')
            except ValueError:
                data['job_card_date'] = pd.to_datetime(data['job_card_date'], dayfirst=True)
        
        return data
    except Exception as e:
        st.error(translate_text(f"Error processing data: {str(e)}", selected_lang_code))
        return None

def create_time_series(data):
    """Create weekly time series."""
    data = data.copy()
    data = data.rename(columns={"job_card_date": "date", "invoice_line_text": "spare_part"})
    data_indexed = data.set_index('date')
    weekly_data = data_indexed[['spare_part']].resample('W').count()
    return weekly_data

def train_test_split(data, test_size=16):
    """Split data into train and test sets."""
    train = data[:-test_size]
    test = data[-test_size:]
    return train, test

def main():
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.write(translate_text("""
    ## Business Case: Vendor Demand Forecast
    This application forecasts demands for each vendor of a particular part based on the percentage of positive reviews for that vendor.
    """, selected_lang_code))
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(translate_text("Upload CSV file", selected_lang_code), type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        data = load_and_validate_data(df)

        if data is None:
            return

        # Sentiment Analysis
        analyzer = SentimentIntensityAnalyzer()
        data['Sentiment'] = data['review_text'].apply(lambda review: 'Positive' if analyzer.polarity_scores(review)['compound'] >= 0.05 else 'Negative')

        # Calculate percentage of positive reviews for each vendor
        vendor_sentiment = data.groupby('supplier_name')['Sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100
        vendor_sentiment['Positive_Percentage'] = vendor_sentiment['Positive']

        # Dropdown menus for selecting vendor and spare part
        vendors = data['supplier_name'].unique()
        selected_vendor = st.selectbox(translate_text("Select Vendor", selected_lang_code), vendors)
        spare_parts = data[data['supplier_name'] == selected_vendor]['invoice_line_text'].unique()
        selected_spare_part = st.selectbox(translate_text("Select Spare Part", selected_lang_code), spare_parts)

        # Filter data based on selected vendor and spare part
        vendor_data = data[(data['supplier_name'] == selected_vendor) & (data['invoice_line_text'] == selected_spare_part)]
        weekly_data = create_time_series(vendor_data)
        train_data, test_data = train_test_split(weekly_data)

        # Holt-Winters Forecast
        hw_model = ExponentialSmoothing(train_data['spare_part'], trend='add', seasonal='add', seasonal_periods=26).fit()
        hw_predictions = hw_model.forecast(len(test_data))

        fig = px.line(title=translate_text(f"Holt-Winters Forecast for {selected_vendor} - {selected_spare_part}", selected_lang_code))
        fig.add_scatter(x=train_data.index, y=train_data['spare_part'], mode='lines+markers', name=translate_text('Train', selected_lang_code), line=dict(color='blue'))
        fig.add_scatter(x=test_data.index, y=test_data['spare_part'], mode='lines+markers', name=translate_text('Test', selected_lang_code), line=dict(color='yellow'))
        fig.add_scatter(x=test_data.index, y=hw_predictions, mode='lines+markers', name=translate_text('Forecast', selected_lang_code), line=dict(color='green'))

        st.plotly_chart(fig, use_container_width=True)

        # Display positive review percentage
        positive_percentage = vendor_sentiment.loc[selected_vendor, 'Positive_Percentage']
        st.write(translate_text(f"Percentage of Positive Reviews: {positive_percentage:.2f}%", selected_lang_code))

if __name__ == "__main__":
    main()