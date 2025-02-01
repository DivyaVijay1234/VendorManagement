import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from chatbot import InventoryBot

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

def display_forecast_plot(response):
    fig = px.line(title="Forecast Comparison")
    
    # Add training data
    fig.add_scatter(
        x=response['data']['train'].index,
        y=response['data']['train']['demand'],
        mode='lines+markers',
        name='Train',
        line=dict(color='blue')
    )
    
    # Add test data
    fig.add_scatter(
        x=response['data']['test'].index,
        y=response['data']['test']['demand'],
        mode='lines+markers',
        name='Test',
        line=dict(color='yellow')
    )
    
    # Add Holt-Winters forecast
    fig.add_scatter(
        x=response['data']['test'].index,
        y=response['data']['hw_pred'],
        mode='lines+markers',
        name='Holt-Winters',
        line=dict(color='green')
    )
    
    # Add SARIMA forecast
    fig.add_scatter(
        x=response['data']['test'].index,
        y=response['data']['sarima_pred'],
        mode='lines+markers',
        name='SARIMA',
        line=dict(color='red')
    )
    
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Demand",
        hovermode='x unified'
    )
    return fig

def display_part_analysis(response):
    # Weekly demand plot
    fig = px.line(
        response['data']['weekly_data'].reset_index(),
        x='date',
        y='demand',
        title=f"Weekly Demand for {response['part_name']}",
        labels={'date': 'Week', 'demand': 'Count'},
        markers=True
    )
    
    if 'train' in response['data']:
        # Create forecast plot
        fig = px.line(title=f"Forecasts for {response['part_name']}")
        fig.add_scatter(x=response['data']['train'].index, 
                       y=response['data']['train']['demand'],
                       mode='lines+markers', name='Train', line=dict(color='blue'))
        fig.add_scatter(x=response['data']['test'].index,
                       y=response['data']['test']['demand'],
                       mode='lines+markers', name='Test', line=dict(color='yellow'))
        fig.add_scatter(x=response['data']['test'].index,
                       y=response['data']['hw_pred'],
                       mode='lines+markers', name='Holt-Winters', line=dict(color='green'))
        fig.add_scatter(x=response['data']['test'].index,
                       y=response['data']['sarima_pred'],
                       mode='lines+markers', name='SARIMA', line=dict(color='red'))
    return fig

def main():
    st.title('AI Inventory Assistant')
    
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
    
    st.write("""
    ### How can I help you today?
    You can ask me questions like:
    - "Forecast demand for next 4 weeks"
    - "Show me demand for Engine Oil"
    - "What's the trend for brake pads?"
    - "Compare forecast models"
    """)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "figure" in message:
                st.plotly_chart(message["figure"], use_container_width=True)
                if "metrics" in message:
                    st.write("""
                    - **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted demand.
                    - **Root Mean Squared Error (RMSE):** Similar to MAE but penalizes larger errors more heavily.
                    """)
                    metrics = message["metrics"]
                    st.write(f"- Holt-Winters MAE: {metrics['hw_mae']:.2f}")
                    st.write(f"- Holt-Winters RMSE: {metrics['hw_rmse']:.2f}")
                    st.write(f"- SARIMA MAE: {metrics['sarima_mae']:.2f}")
                    st.write(f"- SARIMA RMSE: {metrics['sarima_rmse']:.2f}")
    
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
                    fig = display_forecast_plot(response)
                    message = {
                        "role": "assistant",
                        "content": "Here's the forecast comparison:",
                        "figure": fig,
                        "metrics": response['metrics']
                    }
                elif response['type'] == 'part_analysis':
                    fig = display_part_analysis(response)
                    message = {
                        "role": "assistant",
                        "content": f"Analysis for {response['part_name']}",
                        "figure": fig
                    }
                elif response['type'] == 'part_not_found':
                    st.markdown("""
                        <style>
                            .unique-items {
                                height: 300px;
                                overflow-y: scroll;
                                padding: 10px;
                                background-color: #1e1e1e;
                                color: #ffffff;
                                border-radius: 5px;
                                border: 1px solid #ffffff;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    unique_items_html = "<div class='unique-items'>"
                    for item in response['unique_parts']:
                        unique_items_html += f"<p>{item}</p>"
                    unique_items_html += "</div>"
                    st.markdown(unique_items_html, unsafe_allow_html=True)
                    
                    message = {
                        "role": "assistant",
                        "content": response['message']
                    }
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.chat_history.append(message)
            else:
                st.write(response)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })

if __name__ == "__main__":
    main() 