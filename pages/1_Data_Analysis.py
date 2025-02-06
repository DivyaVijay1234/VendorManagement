import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from utils.style import apply_common_style
from utils.translation import translate_text, get_language_code

# Configure page settings (must be the first Streamlit command)
st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")

# Apply common styling
st.markdown(apply_common_style(), unsafe_allow_html=True)

# Sidebar for language selection
languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Urdu', 'Gujarati', 'Punjabi', 'Malayalam', 'Odia', 'Kannada', 'Assamese', 'Maithili', 'Sanskrit']
selected_language = st.sidebar.selectbox("Select Language", languages)
selected_lang_code = get_language_code(selected_language)

# Wrap the title in the header div
st.markdown(f'<div class="header"><h1>{translate_text("Data Analysis Dashboard", selected_lang_code)}</h1></div>', unsafe_allow_html=True)

def load_and_validate_data(df):
    """Load and validate data with flexible column requirements."""
    # Check for minimum required columns
    minimum_required_cols = ['job_card_date', 'invoice_line_text']
    missing_cols = [col for col in minimum_required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(translate_text(f"Missing essential columns: {missing_cols}", selected_lang_code))
        return None
        
    try:
        data = df[pd.notnull(df.invoice_line_text)].reset_index(drop=True)
        
        # Convert job_card_date to datetime with flexible format handling
        try:
            # First try the dd-mm-yy format
            data['job_card_date'] = pd.to_datetime(data['job_card_date'], format='%d-%m-%y')
        except ValueError:
            try:
                # Then try dd-mm-yyyy format
                data['job_card_date'] = pd.to_datetime(data['job_card_date'], format='%d-%m-%Y')
            except ValueError:
                # If both fail, try automatic parsing with dayfirst=True
                data['job_card_date'] = pd.to_datetime(data['job_card_date'], dayfirst=True)
        
        # If additional columns exist, use them for enhanced analysis
        if 'current_km_reading' in df.columns:
            data = data[data.current_km_reading <= 100000].reset_index(drop=True)
            
        return data
    except Exception as e:
        st.error(translate_text(f"Error processing data: {str(e)}", selected_lang_code))
        return None

def create_time_series(data):
    """Create weekly time series."""
    data = data.copy()
    # Rename columns to standardize
    data = data.rename(columns={
        "job_card_date": "date", 
        "invoice_line_text": "spare_part"
    })
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
    ## Business Case: Inventory Management
    Managing spare parts inventory in service centers is a challenge due to fluctuating demand. This application analyzes demand trends 
    and uses predictive models to help service centers align inventory with demand, achieving Just-in-Time (JIT) standards.
    """, selected_lang_code))
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(translate_text("Upload CSV file", selected_lang_code), type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        data = load_and_validate_data(df)

        if data is None:
            return

        # Data Preview section
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.header(translate_text("Data Preview", selected_lang_code))
        with st.expander(translate_text("View Dataset Information", selected_lang_code)):
            st.write(translate_text("First few rows of the dataset:", selected_lang_code))
            st.dataframe(df.head())
            st.write(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
            st.write("Column Names:", df.columns.tolist())
        st.markdown('</div>', unsafe_allow_html=True)

        # Conditional EDA based on available columns
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.header(translate_text("Exploratory Data Analysis", selected_lang_code))

        # Basic analysis (always available)
        with st.expander(translate_text("Unique Items in Invoice", selected_lang_code)):
            display_unique_items(data)

        # Enhanced analysis (if additional columns are available)
        if 'current_km_reading' in data.columns:
            with st.expander(translate_text("KM Reading Distribution", selected_lang_code)):
                display_km_reading_distribution(data)

        with st.expander(translate_text("Top Spare Parts Analysis", selected_lang_code)):
            display_top_spare_parts(data)

        # Time Series Analysis (always available)
        display_time_series_analysis(data)

        # Forecasting (always available)
        display_forecasting_analysis(data)

def display_unique_items(data):
    """Display unique items analysis."""
    st.markdown("""
        <style>
            .unique-items {
                height: 300px;
                overflow-y: scroll;
                padding: 10px;
                background-color: rgba(2, 12, 27, 0.7);
                color: #ffffff;
                border-radius: 5px;
                border: 1px solid #1e2d3d;
            }
        </style>
    """, unsafe_allow_html=True)
    unique_items_html = "<div class='unique-items'>"
    for item in data.invoice_line_text.unique():
        unique_items_html += f"<p>{item}</p>"
    unique_items_html += "</div>"
    st.markdown(unique_items_html, unsafe_allow_html=True)

def display_km_reading_distribution(data):
    """Display KM reading distribution analysis."""
    fig = plt.figure(figsize=(10, 7))
    plt.boxplot(data.current_km_reading)
    plt.title(translate_text("Current KM Reading Distribution", selected_lang_code))
    plt.ylabel(translate_text("Kilometers", selected_lang_code))
    st.pyplot(fig)
    st.write(translate_text("""
    This boxplot shows the distribution of current kilometer readings for vehicles. The majority of readings fall below 100,000 km, 
    as higher readings are considered outliers for service purposes.
    """, selected_lang_code))

def display_top_spare_parts(data):
    """Display top spare parts analysis."""
    st.subheader(translate_text("Top 20 Spare Parts", selected_lang_code))
    fig = plt.figure(figsize=(15, 10))
    sns.countplot(data=data, x='invoice_line_text',
                  order=data.invoice_line_text.value_counts().index[:20])
    plt.xticks(rotation=90)
    plt.title(translate_text("Most Frequently Used Spare Parts", selected_lang_code))
    plt.xlabel(translate_text("Spare Part", selected_lang_code))
    plt.ylabel(translate_text("Count", selected_lang_code))
    st.pyplot(fig)
    st.write(translate_text("""
    This bar chart shows the 20 most frequently used spare parts. 
    Understanding the demand for these items helps optimize inventory levels.
    """, selected_lang_code))

def display_time_series_analysis(data):
    """Display time series analysis."""
    st.header(translate_text("Time Series Analysis", selected_lang_code))
    weekly_data = create_time_series(data)

    with st.expander(translate_text("Weekly Demand Analysis", selected_lang_code)):
        fig = px.line(
            weekly_data.reset_index(),
            x='date',
            y='spare_part',
            title=translate_text("Weekly Demand for Spare Parts", selected_lang_code),
            labels={
                'date': translate_text('Week', selected_lang_code),
                'spare_part': translate_text('Demand Count', selected_lang_code)
            },
            markers=True
        )
        fig.update_traces(hovertemplate=translate_text("Week: %{x}<br>Demand: %{y}<extra></extra>", selected_lang_code))
        st.plotly_chart(fig, use_container_width=True)
        st.write(translate_text("""
        This graph shows the weekly demand for spare parts over time. Peaks represent weeks of high demand, 
        while troughs show lower demand. Hover over a point to view the exact demand value for that week.
        """, selected_lang_code))

    with st.expander(translate_text("Rolling Mean Analysis", selected_lang_code)):
        weekly_data['4W_MA'] = weekly_data['spare_part'].rolling(4).mean()
        fig = px.line(
            weekly_data.reset_index(),
            x='date',
            y=['spare_part', '4W_MA'],
            labels={'date': translate_text('Week', selected_lang_code), 'value': translate_text('Count', selected_lang_code)},
            title=translate_text("Weekly Demand and Rolling Mean (4-Week)", selected_lang_code),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(translate_text("""
        The rolling mean smooths the weekly demand to highlight trends. It averages the demand over a 4-week window, 
        making it easier to observe sustained increases or decreases in demand.
        """, selected_lang_code))

    with st.expander(translate_text("Seasonal Decomposition", selected_lang_code)):
        decomposition = seasonal_decompose(weekly_data['spare_part'], period=4)
        fig = decomposition.plot()
        st.pyplot(fig)
        st.write(translate_text("""
        This decomposition breaks the demand data into three components:
        - **Trend**: Long-term movement in demand.
        - **Seasonal**: Regular patterns over weeks.
        - **Residual**: Random fluctuations.
        Understanding these components helps forecast future demand more accurately.
        """, selected_lang_code))

def display_forecasting_analysis(data):
    """Display forecasting analysis."""
    st.header(translate_text("Time Series Forecasting", selected_lang_code))
    
    # Create time series data
    weekly_data = create_time_series(data)
    train_data, test_data = train_test_split(weekly_data)

    with st.expander(translate_text("Holt-Winters Forecast", selected_lang_code)):
        hw_model = ExponentialSmoothing(
            train_data['spare_part'], 
            trend='mul', 
            seasonal='add', 
            seasonal_periods=26
        ).fit()
        hw_predictions = hw_model.forecast(len(test_data))

        fig = px.line(title=translate_text("Holt-Winters Forecast", selected_lang_code))
        fig.add_scatter(x=train_data.index, y=train_data['spare_part'], mode='lines+markers', 
                       name=translate_text('Train', selected_lang_code), line=dict(color='blue'))
        fig.add_scatter(x=test_data.index, y=test_data['spare_part'], mode='lines+markers', 
                       name=translate_text('Test', selected_lang_code), line=dict(color='yellow'))
        fig.add_scatter(x=test_data.index, y=hw_predictions, mode='lines+markers', 
                       name=translate_text('Forecast', selected_lang_code), line=dict(color='green'))

        st.plotly_chart(fig, use_container_width=True)
        st.write(translate_text("""
        This forecast uses the Holt-Winters method, which accounts for trends and seasonality to predict future demand. 
        The model predicts demand for the test period based on training data.
        """, selected_lang_code))

    with st.expander(translate_text("SARIMA Forecast", selected_lang_code)):
        sarima_model = SARIMAX(
            train_data['spare_part'], 
            order=(5, 1, 1), 
            seasonal_order=(1, 0, 0, 12)
        ).fit()
        sarima_predictions = sarima_model.predict(
            start=len(train_data),
            end=len(train_data) + len(test_data) - 1
        )

        fig = px.line(title=translate_text("SARIMA Forecast", selected_lang_code))
        fig.add_scatter(x=train_data.index, y=train_data['spare_part'], mode='lines+markers', 
                       name=translate_text('Train', selected_lang_code), line=dict(color='blue'))
        fig.add_scatter(x=test_data.index, y=test_data['spare_part'], mode='lines+markers', 
                       name=translate_text('Test', selected_lang_code), line=dict(color='yellow'))
        fig.add_scatter(x=test_data.index, y=sarima_predictions, mode='lines+markers', 
                       name=translate_text('Forecast', selected_lang_code), line=dict(color='green'))

        st.plotly_chart(fig, use_container_width=True)
        st.write(translate_text("""
        The SARIMA model captures seasonal and non-seasonal trends in the data to generate demand forecasts. 
        It is particularly effective for datasets with clear seasonality and periodicity.
        """, selected_lang_code))

    # Metrics section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.header(translate_text("Model Evaluation Metrics", selected_lang_code))
    hw_mae = mean_absolute_error(test_data['spare_part'], hw_predictions)
    hw_rmse = np.sqrt(mean_squared_error(test_data['spare_part'], hw_predictions))
    sarima_mae = mean_absolute_error(test_data['spare_part'], sarima_predictions)
    sarima_rmse = np.sqrt(mean_squared_error(test_data['spare_part'], sarima_predictions))
    
    st.write(translate_text("""
    - **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted demand.
    - **Root Mean Squared Error (RMSE):** Similar to MAE but penalizes larger errors more heavily.
    """, selected_lang_code))
    st.write(translate_text(f"- Holt-Winters MAE: {hw_mae:.2f}", selected_lang_code))
    st.write(translate_text(f"- Holt-Winters RMSE: {hw_rmse:.2f}", selected_lang_code))
    st.write(translate_text(f"- SARIMA MAE: {sarima_mae:.2f}", selected_lang_code))
    st.write(translate_text(f"- SARIMA RMSE: {sarima_rmse:.2f}", selected_lang_code))
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
