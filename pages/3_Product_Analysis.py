import streamlit as st
import pandas as pd
import spacy

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

class SentimentAnalysis:
    def __init__(self):
        self.context = {}

    def set_context(self, df):
        required_cols = ['Product Name', 'Supplier', 'Sentiment', 'Review Text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Missing required columns: {missing_cols}"
        
        self.context['data'] = df.dropna(subset=['Sentiment']).reset_index(drop=True)
        return "Data loaded successfully!"

    def get_products(self):
        if 'data' not in self.context:
            return []
        return sorted(self.context['data']['Product Name'].unique())

    def transform_review(self, review):
        """Converts first-person language to objective language using rule-based replacement."""
        first_person_map = {
            "I": "The user",
            "me": "the user",
            "my": "the user's",
            "we": "customers",
            "us": "customers",
            "our": "customers'"
        }
        
        doc = nlp(review)
        transformed_tokens = [first_person_map.get(token.text, token.text) for token in doc]
        return ' '.join(transformed_tokens).strip()

    def analyze_product(self, product):
        if 'data' not in self.context:
            return "Please upload data first."
        
        df = self.context['data']
        product_df = df[df['Product Name'] == product]
        positive_reviews = product_df[product_df['Sentiment'] == 'Positive']['Review Text'].tolist()
        negative_reviews = product_df[product_df['Sentiment'] == 'Negative']['Review Text'].tolist()
        
        formatted_positive_reviews = ', '.join([self.transform_review(review) for review in positive_reviews[:3]]) if positive_reviews else 'No positive reviews available'
        formatted_negative_reviews = ', '.join([self.transform_review(review) for review in negative_reviews[:3]]) if negative_reviews else 'No negative reviews available'
        
        report = f"""
        **Product Analysis: {product}**
        ---
        - **Advantages:** {formatted_positive_reviews}
        - **Disadvantages:** {formatted_negative_reviews}
        """
        return report

# Streamlit App
st.title("Product Analysis")

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalysis()

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        result = st.session_state.analyzer.set_context(df)
        if result == "Data loaded successfully!":
            st.success(result)
            products = st.session_state.analyzer.get_products()
            if products:
                selected_product = st.selectbox("Select a product to analyze:", products)
                if selected_product:
                    st.write(st.session_state.analyzer.analyze_product(selected_product))
        else:
            st.error(result)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
