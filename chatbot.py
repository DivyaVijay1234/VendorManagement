import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from utils.price_scraper import IndiaMArtScraper

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
        
        # Calculate metrics
        hw_mae = mean_absolute_error(test_data['demand'], hw_predictions)
        hw_rmse = np.sqrt(mean_squared_error(test_data['demand'], hw_predictions))
        sarima_mae = mean_absolute_error(test_data['demand'], sarima_predictions)
        sarima_rmse = np.sqrt(mean_squared_error(test_data['demand'], sarima_predictions))
        
        return {
            'type': 'forecast',
            'data': {
                'train': train_data,
                'test': test_data,
                'hw_pred': hw_predictions,
                'sarima_pred': sarima_predictions
            },
            'metrics': {
                'hw_mae': hw_mae,
                'hw_rmse': hw_rmse,
                'sarima_mae': sarima_mae,
                'sarima_rmse': sarima_rmse
            }
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
                    # Holt-Winters forecast with adjusted parameters
                    hw_model_part = ExponentialSmoothing(
                        train_part['demand'],
                        trend='add',
                        seasonal='add',
                        seasonal_periods=seasonal_periods,
                        initialization_method='estimated'
                    ).fit()
                    hw_pred_part = hw_model_part.forecast(len(test_part))
                except:
                    # Fallback to simple exponential smoothing if Holt-Winters fails
                    hw_model_part = ExponentialSmoothing(
                        train_part['demand'],
                        trend=None,
                        seasonal=None
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
                
                return {
                    'type': 'part_analysis',
                    'data': {
                        'weekly_data': part_weekly,
                        'train': train_part,
                        'test': test_part,
                        'hw_pred': hw_pred_part,
                        'sarima_pred': sarima_pred_part
                    },
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

    def process_query(self, query):
        if 'data' not in self.context:
            return "Please upload data first."
            
        query = query.lower().strip()
        
        if 'forecast' in query or 'compare' in query:
            return self.handle_forecast_request(query)
        elif 'demand' in query or 'trend' in query:
            # Extract part name after "for" or "of"
            parts = query.split()
            for i, word in enumerate(parts):
                if word in ['for', 'of'] and i + 1 < len(parts):
                    part_name = ' '.join(parts[i+1:])
                    return self.handle_part_query(part_name.strip())
            return "Please specify a part name (e.g., 'Show demand for Engine Oil')"
        else:
            return """I can help you with:
                   - Forecast demand
                   - Show demand for [part name]
                   - Show trend for [part name]
                   - Compare models"""

    def set_context(self, df):
        """Load and validate data."""
        required_cols = ['invoice_date', 'job_card_date', 'business_partner_name',
                        'vehicle_no', 'vehicle_model', 'current_km_reading', 'invoice_line_text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Missing required columns: {missing_cols}"
        try:
            data = df[pd.notnull(df.invoice_line_text)].reset_index(drop=True)
            data = data[data.current_km_reading <= 100000].reset_index(drop=True)
            data['job_card_date'] = pd.to_datetime(data['job_card_date'], format='%d-%m-%y')
            self.context['data'] = data
            return "Data loaded successfully!"
        except Exception as e:
            return f"Error processing data: {str(e)}" 