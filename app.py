import streamlit as st

def main():
    st.title('Inventory Management System')
    
    st.write("""
    ## Welcome to the Inventory Management System
    
    This application provides two main features:
    
    ### 📊 Data Analysis Dashboard
    Access comprehensive inventory analysis tools including:
    - Data validation and preview
    - Exploratory Data Analysis
    - Time Series Analysis
    - Demand Forecasting
    
    ### 🤖 AI Assistant
    Interact with our AI-powered chatbot to:
    - Get demand predictions
    - Analyze specific parts
    - Understand inventory trends
    - Compare forecasting models
    
    Select a page from the sidebar to get started!
    """)

if __name__ == "__main__":
    main() 