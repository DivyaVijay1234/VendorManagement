import streamlit as st
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from chatbot import InventoryBot
from utils.style import apply_common_style
from utils.translation import translate_text, get_language_code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import os
import importlib

# Add the pages directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
pages_dir = os.path.dirname(current_dir)
if pages_dir not in sys.path:
    sys.path.append(pages_dir)

# Import using importlib
forecast_analysis = importlib.import_module("pages.5_Forecast_Analysis")
train_random_forest = forecast_analysis.train_random_forest
train_xgboost = forecast_analysis.train_xgboost
calculate_metrics = forecast_analysis.calculate_metrics

# Apply common styling
st.markdown(apply_common_style(), unsafe_allow_html=True)

# Sidebar for language selection
languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Urdu', 'Gujarati', 'Punjabi', 'Malayalam', 'Odia', 'Kannada', 'Assamese', 'Maithili', 'Sanskrit']
selected_language = st.sidebar.selectbox("Select Language", languages)
selected_lang_code = get_language_code(selected_language)

# Wrap the title in the header div
st.markdown(f'<div class="header"><h1>{translate_text("AI Inventory Assistant", selected_lang_code)}</h1></div>', unsafe_allow_html=True)

def prepare_ml_features(data):
    """Prepare features for ML models."""
    data = data.copy()
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['quarter'] = data.index.quarter
    data['week'] = data.index.isocalendar().week
    
    # Create a dummy item_code since we're dealing with a single part
    data['item_code'] = 0
    
    X = data[['year', 'month', 'day', 'dayofweek', 'quarter', 'week', 'item_code']]
    y = data['demand']
    return X, y

def display_forecast_plot(response):
    """Display forecast plot with all models."""
    if not isinstance(response, dict) or 'part_name' not in response:
        st.error("Invalid response format")
        return None, None
        
    # Create the plot with existing models first
    fig = px.line(title=translate_text(f"Forecasts for {response.get('part_name', 'Unknown Part')}", selected_lang_code))
    
    # Check if required data exists
    if not all(key in response.get('data', {}) for key in ['train', 'test', 'hw_pred', 'sarima_pred', 'rf_pred', 'xgb_pred']):
        st.error("Missing required forecast data")
        return None, None
    
    # Add training data
    fig.add_scatter(
        x=response['data']['train'].index, 
        y=response['data']['train']['demand'],
        mode='lines+markers', 
        name=translate_text('Train', selected_lang_code), 
        line=dict(color='blue')
    )
    
    # Add test data
    fig.add_scatter(
        x=response['data']['test'].index,
        y=response['data']['test']['demand'],
        mode='lines+markers', 
        name=translate_text('Test', selected_lang_code), 
        line=dict(color='yellow')
    )
    
    # Add Holt-Winters predictions
    fig.add_scatter(
        x=response['data']['test'].index,
        y=response['data']['hw_pred'],
        mode='lines+markers', 
        name=translate_text('Holt-Winters', selected_lang_code), 
        line=dict(color='green')
    )
    
    # Add SARIMA predictions
    fig.add_scatter(
        x=response['data']['test'].index,
        y=response['data']['sarima_pred'],
        mode='lines+markers', 
        name=translate_text('SARIMA', selected_lang_code), 
        line=dict(color='red')
    )
    
    # Add Random Forest predictions
    fig.add_scatter(
        x=response['data']['test'].index,
        y=response['data']['rf_pred'],
        mode='lines+markers', 
        name=translate_text('Random Forest', selected_lang_code), 
        line=dict(color='purple')
    )
    
    # Add XGBoost predictions
    fig.add_scatter(
        x=response['data']['test'].index,
        y=response['data']['xgb_pred'],
        mode='lines+markers', 
        name=translate_text('XGBoost', selected_lang_code), 
        line=dict(color='orange')
    )
    
    # Update layout
    fig.update_layout(
        title=translate_text(f"Forecasts for {response['part_name']}", selected_lang_code),
        xaxis_title=translate_text("Week", selected_lang_code),
        yaxis_title=translate_text("Demand", selected_lang_code),
        hovermode='x unified'
    )
    
    return fig, response.get('metrics')

def display_part_analysis(response):
    col1, col2 = st.columns([2, 1])
    fig = None
    
    with col1:
        if response['type'] in ['parts_analysis', 'unique_parts']:
            st.subheader(response.get('title', 'Parts Analysis'))
            if response['type'] == 'parts_analysis':
                fig = px.bar(
                    x=response['data'].index,
                    y=response['data'].values,
                    title=response['title'],
                    labels={'x': 'Spare Part', 'y': 'Count'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="parts_analysis")
            else:
                st.write(response['message'])
                st.write(response['data'])
                
        elif response['type'] in ['weekly_analysis', 'rolling_analysis']:
            fig = px.line(
                response['data'].reset_index(),
                x='date',
                y=response['data'].columns,
                title=response['title']
            )
            st.plotly_chart(fig, use_container_width=True, key="weekly_analysis")
            
        elif response['type'] == 'part_analysis':
            fig = px.line(
                response['data']['weekly_data'].reset_index(),
                x='date',
                y='demand',
                title=f"Weekly Demand for {response['part_name']}",
                labels={'date': 'Week', 'demand': 'Count'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True, key="part_demand")
            if 'train' in response['data']:
                fig, metrics = display_forecast_plot(response)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="part_forecast")
    
    with col2:
        if 'price_data' in response and response['price_data'] and 'products' in response['price_data']:
            st.subheader("Market Price Analysis")
            price_data = response['price_data']
            
            st.metric(
                "Average Market Price",
                f"₹{price_data['avg_price']:,.2f}",
                delta=None
            )
            
            st.write("Price Range:")
            st.progress((price_data['avg_price'] - price_data['min_price']) / 
                       (price_data['max_price'] - price_data['min_price']))
            st.write(f"₹{price_data['min_price']:,.2f} - ₹{price_data['max_price']:,.2f}")
            
            st.subheader("Top Suppliers")
            for product in price_data['products']:
                with st.expander(f"{product['company']}"):
                    st.write(f"**Product:** {product['title']}")
                    st.write(f"**Price:** ₹{product['price']:,.2f}{product['unit']}")
                    if product['rating']:
                        stars = "⭐" * int(round(product['rating']))
                        empty_stars = "☆" * (5 - int(round(product['rating'])))
                        st.write(f"Rating: {stars}{empty_stars} ({product['rating']}/5)")
                        st.write(f"Number of Ratings: {product['num_ratings']}")
                    st.write(f"**Location:** {product['location']}")
    
    return fig

def main():
    # Main content section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    if 'bot' not in st.session_state:
        st.session_state.bot = InventoryBot()
    
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader("Upload inventory CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            result = st.session_state.bot.set_context(df)
            if result == "Data loaded successfully!":
                st.success(result)
            else:
                st.error(result)
    
    st.write(translate_text("""### How can I help you today?
    You can ask me questions like:
    - "Forecast demand for next 4 weeks"
    - "Show me demand for Engine Oil"
    - "What's the trend for Brake Disc"
    - "Compare forecast models"
    - "Show top 20 parts"
    - "Show top 50 parts"
    - "List unique parts"
    - "Show weekly demand"
    - "Show weekly demand vs rolling mean"
    """, selected_lang_code))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "figure" in message:
                st.plotly_chart(message["figure"], use_container_width=True, key=f"history_{i}")
                if "metrics" in message:
                    st.write("""### Model Performance Metrics:
                    - **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values
                    - **Root Mean Squared Error (RMSE):** Square root of the average squared differences""")
                    metrics = message["metrics"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("#### Traditional Models")
                        st.write(f"- Holt-Winters MAE: {metrics['hw_mae']:.2f}")
                        st.write(f"- Holt-Winters RMSE: {metrics['hw_rmse']:.2f}")
                        st.write(f"- SARIMA MAE: {metrics['sarima_mae']:.2f}")
                        st.write(f"- SARIMA RMSE: {metrics['sarima_rmse']:.2f}")
                    with col2:
                        st.write("#### Machine Learning Models")
                        st.write(f"- Random Forest MAE: {metrics['rf_mae']:.2f}")
                        st.write(f"- Random Forest RMSE: {metrics['rf_rmse']:.2f}")
                        st.write(f"- XGBoost MAE: {metrics['xgb_mae']:.2f}")
                        st.write(f"- XGBoost RMSE: {metrics['xgb_rmse']:.2f}")
    
    user_query = st.chat_input("Ask me about inventory demand predictions:")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            response = st.session_state.bot.process_query(user_query)
            
            if isinstance(response, dict):
                st.write(response.get('message', ''))
                fig = None
                
                if response['type'] == 'forecast':
                    fig, metrics = display_forecast_plot(response)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"forecast_{len(st.session_state.chat_history)}")
                    message = {
                        "role": "assistant",
                        "content": "Here's the forecast comparison:",
                        "figure": fig,
                        "metrics": metrics
                    }
                elif response['type'] == 'part_analysis':
                    fig = display_part_analysis(response)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"part_analysis_{len(st.session_state.chat_history)}")
                    message = {
                        "role": "assistant",
                        "content": f"Analysis for {response['part_name']}",
                        "figure": fig
                    }
                elif response['type'] == 'part_not_found':
                    st.markdown("""<style>.unique-items {height: 300px;overflow-y: scroll;padding: 10px;background-color: #1e1e1e;color: #ffffff;border-radius: 5px;border: 1px solid #ffffff;}</style>""", unsafe_allow_html=True)
                    
                    unique_items_html = "<div class='unique-items'>"
                    for item in response['unique_parts']:
                        unique_items_html += f"<p>{item}</p>"
                    unique_items_html += "</div>"
                    st.markdown(unique_items_html, unsafe_allow_html=True)
                    
                    message = {"role": "assistant", "content": response['message']}
                elif response['type'] in ['parts_analysis', 'unique_parts', 'weekly_analysis', 'rolling_analysis']:
                    fig = display_part_analysis(response)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"analysis_{response['type']}_{len(st.session_state.chat_history)}")
                    message = {"role": "assistant", "content": response['message'], "figure": fig}
                
                if fig:
                    st.session_state.chat_history.append(message)
            else:
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()