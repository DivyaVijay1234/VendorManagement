# Leveraging LLM’S for AI-Driven Demand Prediction

Welcome to the Inventory Management System. This application leverages large language models (LLMs) for AI-driven demand prediction and provides comprehensive tools for data analysis, AI assistance, product analysis, and sentiment analysis.

## Features

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

### 📝 Product Analysis
Perform detailed analysis of product reviews to:
- Understand customer feedback
- Identify advantages and disadvantages of products
- Track product performance
- Generate insights

### 😊 Sentiment Analysis
Measure the sentiment of specific vendors to:
- Understand vendor performance
- Compare sentiment across suppliers
- Track satisfaction trends
- Identify improvement areas

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/inventory-management-system.git
    cd inventory-management-system
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the SpaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to access the application.

## File Structure

- [app.py](http://_vscodecontentref_/1): Main application file that sets up the Streamlit interface and navigation.
- [1_Data_Analysis.py](http://_vscodecontentref_/2): Data analysis dashboard.
- [2_AI_Assistant.py](http://_vscodecontentref_/3): AI assistant for demand prediction and analysis.
- [3_Product_Analysis.py](http://_vscodecontentref_/4): Product analysis based on customer reviews.
- [4_Sentiment_Analysis.py](http://_vscodecontentref_/5): Sentiment analysis of specific vendors.
- [style.py](http://_vscodecontentref_/6): Common styling for the application.

## Contributing

We welcome contributions to improve the Inventory Management System. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](http://_vscodecontentref_/7) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [SpaCy](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)