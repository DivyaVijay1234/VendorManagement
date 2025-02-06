import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.style import apply_common_style
from utils.translation import translate_text, get_language_code
import joblib
import os
from hashlib import md5
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create a cache directory if it doesn't exist
CACHE_DIR = "model_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Remove the set_page_config since this file is being imported as a module
# st.set_page_config(page_title="Forecast Analysis", page_icon="📊", layout="wide")

# Apply common styling
st.markdown(apply_common_style(), unsafe_allow_html=True)

# Sidebar for language selection
languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Urdu', 'Gujarati', 'Punjabi', 'Malayalam', 'Odia', 'Kannada', 'Assamese', 'Maithili', 'Sanskrit']
selected_language = st.sidebar.selectbox("Select Language", languages)
selected_lang_code = get_language_code(selected_language)

def get_data_hash(data):
    """Generate a hash for the input data."""
    return md5(str(data.values.tobytes()).encode()).hexdigest()

with st.sidebar:
    if st.button(translate_text('Clear Model Cache', selected_lang_code)):
        try:
            for file in os.listdir(CACHE_DIR):
                os.remove(os.path.join(CACHE_DIR, file))
            st.cache_resource.clear()
            st.success(translate_text('Cache cleared successfully!', selected_lang_code))
        except Exception as e:
            st.error(translate_text(f'Error clearing cache: {str(e)}', selected_lang_code))

def load_and_preprocess_data(df):
    """Load and preprocess data."""
    # Check for minimum required columns
    minimum_required_cols = ['job_card_date', 'invoice_line_text']
    missing_cols = [col for col in minimum_required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(translate_text(f"Missing essential columns: {missing_cols}", selected_lang_code))
        return None
        
    try:
        data = df.copy()
        
        # Convert job_card_date to datetime
        data['job_card_date'] = pd.to_datetime(data['job_card_date'], format='%d-%m-%Y')
        
        # Create time-based features
        data['year'] = data['job_card_date'].dt.year
        data['month'] = data['job_card_date'].dt.month
        data['day'] = data['job_card_date'].dt.day
        data['dayofweek'] = data['job_card_date'].dt.dayofweek
        data['quarter'] = data['job_card_date'].dt.quarter
        data['week'] = data['job_card_date'].dt.isocalendar().week
        
        # Encode invoice_line_text
        le = LabelEncoder()
        data['item_code'] = le.fit_transform(data['invoice_line_text'])
        
        # Create daily counts for each item
        daily_counts = data.groupby(['job_card_date', 'item_code']).size().reset_index(name='count')
        
        # Create feature matrix with the same index as daily_counts
        X = pd.DataFrame({
            'year': daily_counts['job_card_date'].dt.year,
            'month': daily_counts['job_card_date'].dt.month,
            'day': daily_counts['job_card_date'].dt.day,
            'dayofweek': daily_counts['job_card_date'].dt.dayofweek,
            'quarter': daily_counts['job_card_date'].dt.quarter,
            'week': daily_counts['job_card_date'].dt.isocalendar().week,
            'item_code': daily_counts['item_code']
        })
        
        y = daily_counts['count']
        
        return X, y, data, le, daily_counts
        
    except Exception as e:
        st.error(translate_text(f"Error processing data: {str(e)}", selected_lang_code))
        return None

@st.cache_resource
def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest model."""
    with st.expander(translate_text("Random Forest Model", selected_lang_code)):
        st.write(translate_text("""
        Random Forest is an ensemble learning method that operates by constructing multiple decision trees and outputs the mean 
        prediction of the individual trees. It's effective for both regression and classification tasks.
        """, selected_lang_code))
        
        with st.spinner(translate_text('Training Random Forest model...', selected_lang_code)):
            # Hyperparameter grid
            random_grid = {
                'n_estimators': [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [int(x) for x in np.linspace(5, 30, num=6)],
                'min_samples_split': [2, 5, 10, 15, 100],
                'min_samples_leaf': [1, 2, 5, 10]
            }
            
            rf = RandomForestRegressor()
            rf_random = RandomizedSearchCV(
                estimator=rf,
                param_distributions=random_grid,
                n_iter=100,
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            rf_random.fit(X_train, y_train)
            
            # Predictions and metrics
            y_pred = rf_random.predict(X_test)
            r2_score = rf_random.score(X_test, y_test)
            adj_r2 = 1 - (1 - r2_score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(translate_text('R-squared Score', selected_lang_code), f"{r2_score:.4f}")
            with col2:
                st.metric(translate_text('Adjusted R-squared', selected_lang_code), f"{adj_r2:.4f}")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_pred[500:800], name=translate_text('Predictions', selected_lang_code)))
            fig.add_trace(go.Scatter(y=y_test[500:800], name=translate_text('Actual Values', selected_lang_code)))
            fig.update_layout(title=translate_text('Random Forest: Forecast vs Actual Values', selected_lang_code))
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            importance_df = pd.DataFrame({
                'Feature': ['Year', 'Month', 'Day', 'Day of Week', 'Quarter', 'Week', 'Item Code'],
                'Importance': rf_random.best_estimator_.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        title=translate_text('Random Forest: Feature Importance', selected_lang_code))
            st.plotly_chart(fig, use_container_width=True)
            
            return rf_random, y_pred

@st.cache_resource
def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model."""
    with st.expander(translate_text("XGBoost Model", selected_lang_code)):
        st.write(translate_text("""
        XGBoost is an optimized gradient boosting library. It provides parallel tree boosting and is highly efficient, 
        flexible, and portable.
        """, selected_lang_code))
        
        with st.spinner(translate_text('Training XGBoost model...', selected_lang_code)):
            # Direct parameter tuning instead of RandomizedSearchCV
            xgb = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            
            # Fit the model
            xgb.fit(X_train, y_train)
            
            # Predictions and metrics
            y_pred = xgb.predict(X_test)
            r2_score = xgb.score(X_test, y_test)
            adj_r2 = 1 - (1 - r2_score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(translate_text('R-squared Score', selected_lang_code), f"{r2_score:.4f}")
            with col2:
                st.metric(translate_text('Adjusted R-squared', selected_lang_code), f"{adj_r2:.4f}")
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_pred[500:800], name=translate_text('Predictions', selected_lang_code)))
            fig.add_trace(go.Scatter(y=y_test[500:800], name=translate_text('Actual Values', selected_lang_code)))
            fig.update_layout(title=translate_text('XGBoost: Forecast vs Actual Values', selected_lang_code))
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            importance_df = pd.DataFrame({
                'Feature': ['Year', 'Month', 'Day', 'Day of Week', 'Quarter', 'Week', 'Item Code'],
                'Importance': xgb.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        title=translate_text('XGBoost: Feature Importance', selected_lang_code))
            st.plotly_chart(fig, use_container_width=True)
            
            return xgb, y_pred

# Add this function to calculate all metrics
def calculate_metrics(y_true, y_pred, n_features=7):
    """Calculate multiple performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Adjusted R² formula: 1 - (1-R²)*(n-1)/(n-k-1) where n is sample size and k is number of features
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_features - 1)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r2,
        'Adjusted R-squared': adj_r2
    }

def main():
    st.markdown(f'<div class="header"><h1>{translate_text("Forecast Analysis Dashboard", selected_lang_code)}</h1></div>', unsafe_allow_html=True)
    
    st.write(translate_text("""
    ## Business Case: Demand Forecasting
    This analysis helps predict future demand for spare parts based on historical data. We use advanced machine learning 
    models (Random Forest and XGBoost) to capture patterns and make accurate predictions.
    """, selected_lang_code))
    
    uploaded_file = st.file_uploader(translate_text("Upload CSV file", selected_lang_code), type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            result = load_and_preprocess_data(df)
            
            if result is not None:
                X, y, data, le, daily_counts = result
                
                # Data Overview
                with st.expander(translate_text("Data Overview", selected_lang_code)):
                    st.write(translate_text("First few rows of processed data:", selected_lang_code))
                    st.dataframe(daily_counts.head())
                    st.write(f"Total Records: {len(daily_counts)}")
                    st.write(f"Date Range: {daily_counts['job_card_date'].min()} to {daily_counts['job_card_date'].max()}")
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train models (removed tuned random forest)
                rf_model, rf_pred = train_random_forest(X_train, y_train, X_test, y_test)
                xgb_model, xgb_pred = train_xgboost(X_train, y_train, X_test, y_test)
                
                # Enhanced Model Comparison
                with st.expander(translate_text("Model Comparison", selected_lang_code)):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_test[500:800], 
                                           name=translate_text('Actual Values', selected_lang_code)))
                    fig.add_trace(go.Scatter(y=rf_pred[500:800], 
                                           name=translate_text('Random Forest', selected_lang_code)))
                    fig.add_trace(go.Scatter(y=xgb_pred[500:800], 
                                           name=translate_text('XGBoost', selected_lang_code)))
                    fig.update_layout(title=translate_text('Model Comparison: Forecasts vs Actual Values', selected_lang_code))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate metrics for models
                    rf_metrics = calculate_metrics(y_test, rf_pred)
                    xgb_metrics = calculate_metrics(y_test, xgb_pred)
                    
                    # Create comprehensive performance comparison
                    performance_df = pd.DataFrame({
                        'Metric': ['MAE', 'RMSE', 'R-squared', 'Adjusted R-squared'],
                        'Random Forest': [rf_metrics['MAE'], rf_metrics['RMSE'], 
                                        rf_metrics['R-squared'], rf_metrics['Adjusted R-squared']],
                        'XGBoost': [xgb_metrics['MAE'], xgb_metrics['RMSE'],
                                   xgb_metrics['R-squared'], xgb_metrics['Adjusted R-squared']]
                    })
                    
                    st.subheader(translate_text('Model Performance Comparison', selected_lang_code))
                    st.dataframe(performance_df.style.format({
                        'Random Forest': '{:.4f}',
                        'XGBoost': '{:.4f}'
                    }))
                    
                    # Add explanation of metrics
                    st.write(translate_text("""
                    ### Metric Explanations:
                    - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values. Lower is better.
                    - **RMSE (Root Mean Square Error)**: Square root of the average squared differences. More sensitive to large errors. Lower is better.
                    - **R-squared**: Proportion of variance in the target that's predictable from the features. Higher is better (max 1.0).
                    - **Adjusted R-squared**: R-squared adjusted for the number of features. Better for comparing models with different numbers of features.
                    """, selected_lang_code))
                    
                    # Add insights about model performance
                    st.write(translate_text("""
                    ### Key Insights:
                    - MAE and RMSE provide a sense of prediction error in the original units
                    - R-squared and Adjusted R-squared show how well the model explains the variance in the data
                    - Lower MAE/RMSE and higher R-squared values indicate better model performance
                    - Consider the trade-off between model complexity and performance improvement
                    """, selected_lang_code))
                    
                    # Add visualization of metrics
                    metrics_comparison = performance_df.melt(id_vars=['Metric'], 
                                                           var_name='Model', 
                                                           value_name='Value')
                    
                    fig = px.bar(metrics_comparison, 
                                x='Model', 
                                y='Value', 
                                color='Metric', 
                                barmode='group',
                                title=translate_text('Model Performance Metrics Comparison', selected_lang_code))
                    st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(translate_text(f"Error: {str(e)}", selected_lang_code))
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 