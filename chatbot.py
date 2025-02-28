import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from utils.price_scraper import IndiaMArtScraper
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

class InventoryBot:
    def __init__(self):
        self.context = {}
        self.price_scraper = IndiaMArtScraper()

    def create_time_series(self, data):
        """Create weekly time series."""
        data = data.copy()
        data = data.rename(columns={"job_card_date": "date", "invoice_line_text": "demand"})
        data_indexed = data.set_index('date')
        weekly_data = data_indexed[['demand']].resample('W').count()
        return weekly_data

    def train_test_split(self, data, test_size=16):
        """Split data into train and test sets."""
        train = data[:-test_size]
        test = data[-test_size:]
        return train, test

    def handle_forecast_request(self, query):
        if 'data' not in self.context:
            return "Please upload data first."
        
        # Create time series
        weekly_data = self.create_time_series(self.context['data'])
        train_data, test_data = self.train_test_split(weekly_data)
        
        # Holt-Winters forecast
        hw_model = ExponentialSmoothing(
            train_data['demand'],
            trend='add',
            seasonal='add',
            seasonal_periods=26
        ).fit()
        hw_predictions = hw_model.forecast(len(test_data))
        
        # SARIMA forecast
        sarima_model = SARIMAX(
            train_data['demand'],
            order=(5,1,1),
            seasonal_order=(1,0,0,12)
        ).fit(disp=False)
        sarima_predictions = sarima_model.predict(
            start=len(train_data),
            end=len(train_data)+len(test_data)-1
        )
        
        # Prepare data for ML models
        X_train = pd.DataFrame({
            'year': train_data.index.year,
            'month': train_data.index.month,
            'day': train_data.index.day,
            'dayofweek': train_data.index.dayofweek,
            'quarter': train_data.index.quarter,
            'week': train_data.index.isocalendar().week
        })
        
        X_test = pd.DataFrame({
            'year': test_data.index.year,
            'month': test_data.index.month,
            'day': test_data.index.day,
            'dayofweek': test_data.index.dayofweek,
            'quarter': test_data.index.quarter,
            'week': test_data.index.isocalendar().week
        })
        
        y_train = train_data['demand']
        y_test = test_data['demand']
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        
        # Train XGBoost
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_predictions = xgb_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'hw_mae': mean_absolute_error(test_data['demand'], hw_predictions),
            'hw_rmse': np.sqrt(mean_squared_error(test_data['demand'], hw_predictions)),
            'sarima_mae': mean_absolute_error(test_data['demand'], sarima_predictions),
            'sarima_rmse': np.sqrt(mean_squared_error(test_data['demand'], sarima_predictions)),
            'rf_mae': mean_absolute_error(test_data['demand'], rf_predictions),
            'rf_rmse': np.sqrt(mean_squared_error(test_data['demand'], rf_predictions)),
            'xgb_mae': mean_absolute_error(test_data['demand'], xgb_predictions),
            'xgb_rmse': np.sqrt(mean_squared_error(test_data['demand'], xgb_predictions))
        }
        
        return {
            'type': 'forecast',
            'part_name': 'All Parts',
            'data': {
                'train': train_data,
                'test': test_data,
                'hw_pred': hw_predictions,
                'sarima_pred': sarima_predictions,
                'rf_pred': rf_predictions,
                'xgb_pred': xgb_predictions
            },
            'metrics': metrics
        }

    def handle_part_query(self, part_name):
        if 'data' not in self.context:
            return "Please upload data first."
        
        data = self.context['data']
        part_data = data[data['invoice_line_text'].str.upper() == part_name.upper()].copy()
        
        if len(part_data) == 0:
            unique_parts = sorted(data['invoice_line_text'].unique())
            return {
                'type': 'part_not_found',
                'message': f"No data found for '{part_name}'. Here are all available parts:",
                'unique_parts': unique_parts
            }
        
        # Create weekly time series for the part
        part_data = part_data.rename(columns={"job_card_date": "date"})
        part_weekly = part_data.set_index('date').resample('W').size().to_frame('demand')
        
        # Get price data from IndiaMart
        price_data = self.price_scraper.get_price_data(part_name)
        
        if len(part_weekly) > 0:
            if len(part_weekly) >= 26:
                train_part, test_part = self.train_test_split(part_weekly)
                
                # Calculate appropriate seasonal period
                seasonal_periods = min(12, len(train_part) // 2)
                
                try:
                    # Holt-Winters forecast
                    hw_model_part = ExponentialSmoothing(
                        train_part['demand'],
                        trend='add',
                        seasonal='add',
                        seasonal_periods=seasonal_periods,
                        initialization_method='estimated'
                    ).fit()
                    hw_pred_part = hw_model_part.forecast(len(test_part))
                    
                    # SARIMA forecast
                    sarima_model_part = SARIMAX(
                        train_part['demand'],
                        order=(5,1,1),
                        seasonal_order=(1,0,0,12)
                    ).fit(disp=False)
                    sarima_pred_part = sarima_model_part.predict(
                        start=len(train_part),
                        end=len(train_part)+len(test_part)-1
                    )
                    
                    # Prepare data for ML models
                    X_train = pd.DataFrame({
                        'year': train_part.index.year,
                        'month': train_part.index.month,
                        'day': train_part.index.day,
                        'dayofweek': train_part.index.dayofweek,
                        'quarter': train_part.index.quarter,
                        'week': train_part.index.isocalendar().week
                    })
                    
                    X_test = pd.DataFrame({
                        'year': test_part.index.year,
                        'month': test_part.index.month,
                        'day': test_part.index.day,
                        'dayofweek': test_part.index.dayofweek,
                        'quarter': test_part.index.quarter,
                        'week': test_part.index.isocalendar().week
                    })
                    
                    y_train = train_part['demand']
                    y_test = test_part['demand']
                    
                    # Train Random Forest
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    rf_pred_part = rf_model.predict(X_test)
                    
                    # Train XGBoost
                    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
                    xgb_model.fit(X_train, y_train)
                    xgb_pred_part = xgb_model.predict(X_test)
                    
                    return {
                        'type': 'part_analysis',
                        'data': {
                            'weekly_data': part_weekly,
                            'train': train_part,
                            'test': test_part,
                            'hw_pred': hw_pred_part,
                            'sarima_pred': sarima_pred_part,
                            'rf_pred': rf_pred_part,
                            'xgb_pred': xgb_pred_part
                        },
                        'price_data': price_data,
                        'part_name': part_name
                    }
                except Exception as e:
                    return {
                        'type': 'part_analysis',
                        'data': {'weekly_data': part_weekly},
                        'price_data': price_data,
                        'part_name': part_name
                    }
            else:
                return {
                    'type': 'part_analysis',
                    'data': {'weekly_data': part_weekly},
                    'price_data': price_data,
                    'part_name': part_name
                }
        return f"No weekly data available for {part_name}"

    def get_price_prediction(self, part_name, demand_forecast):
        # Get current price data
        price_data = self.price_scraper.get_price_data(part_name)
        
        if price_data:
            # Simple price prediction based on demand changes
            current_demand = self.context['data'][self.context['data']['invoice_line_text'] == part_name].shape[0]
            future_demand = demand_forecast.mean()
            
            demand_change_ratio = future_demand / current_demand if current_demand > 0 else 1
            
            # Adjust price based on demand changes (simple linear relationship)
            predicted_price = price_data['avg_price'] * (1 + (demand_change_ratio - 1) * 0.5)
            
            return {
                'current_price': price_data['avg_price'],
                'predicted_price': predicted_price,
                'price_range': {
                    'min': price_data['min_price'],
                    'max': price_data['max_price']
                }
            }
        return None
    
    def handle_top_parts_query(self, query):
        if 'top 50' in query.lower():
            n = 50
        else:
            n = 20
            
        top_parts = self.context['data']['invoice_line_text'].value_counts()[:n]
        
        return {
            'type': 'parts_analysis',
            'data': top_parts,
            'title': f'Top {n} Spare Parts',
            'message': f"Here are the top {n} most frequently used spare parts:"
        }
    
    def handle_unique_parts_query(self):
        unique_parts = sorted(self.context['data']['invoice_line_text'].unique())
        return {
            'type': 'unique_parts',
            'data': unique_parts,
            'message': "Here are all unique spare parts in inventory:"
        }
    
    def handle_weekly_analysis(self, query):
        weekly_data = self.create_time_series(self.context['data'])
        
        if 'rolling' in query.lower():
            weekly_data['4W_MA'] = weekly_data['demand'].rolling(4).mean()
            return {
                'type': 'rolling_analysis',
                'data': weekly_data,
                'title': 'Weekly Demand vs Rolling Mean (4-Week)',
                'message': "Comparing weekly demand with 4-week rolling mean"
            }
        else:
            return {
                'type': 'weekly_analysis',
                'data': weekly_data,
                'title': 'Weekly Spare Parts Demand',
                'message': "Weekly demand analysis for all spare parts"
            }

    def process_query(self, query):
        if 'data' not in self.context:
            return "Please upload data first."
            
        query = query.lower().strip()
        
        if 'top' in query and ('50' in query or '20' in query):
            return self.handle_top_parts_query(query)
        elif 'list' in query and 'unique' in query:
            return self.handle_unique_parts_query()
        elif 'weekly' in query or 'rolling' in query:
            return self.handle_weekly_analysis(query)
        elif 'forecast' in query or 'compare' in query:
            return self.handle_forecast_request(query)
        elif 'demand' in query or 'trend' in query:
            parts = query.split()
            for i, word in enumerate(parts):
                if word in ['for', 'of'] and i + 1 < len(parts):
                    part_name = ' '.join(parts[i+1:])
                    return self.handle_part_query(part_name.strip())
            return "Please specify a part name"
        else:
            return """I can help you with:
                   - Forecast demand
                   - Show demand for [part name]
                   - Show trend for [part name]
                   - Compare models
                   - Show top 20/50 parts
                   - List unique parts
                   - Show weekly demand
                   - Show weekly demand vs rolling mean"""

    def set_context(self, df):
        """Load and validate data."""
        # Check for minimum required columns
        minimum_required_cols = ['job_card_date', 'invoice_line_text']
        missing_cols = [col for col in minimum_required_cols if col not in df.columns]
        
        if missing_cols:
            return f"Missing essential columns: {missing_cols}"
        
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
            
            self.context['data'] = data
            return "Data loaded successfully!"
        except Exception as e:
            return f"Error processing data: {str(e)}"