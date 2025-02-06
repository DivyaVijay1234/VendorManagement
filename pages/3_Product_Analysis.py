import streamlit as st
import pandas as pd
import spacy
from utils.style import apply_common_style
from utils.translation import translate_text, get_language_code

class PartAnalysis:
    def __init__(self):
        self.context = {}

    def set_context(self, df):
        required_cols = ['Part', 'Supplier', 'Advantages', 'Disadvantages']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Missing required columns: {missing_cols}"
        
        self.context['data'] = df.dropna().reset_index(drop=True)
        return "Data loaded successfully!"

    def get_parts(self):
        if 'data' not in self.context:
            return []
        return sorted(self.context['data']['Part'].unique())
    
    def get_suppliers(self, part):
        if 'data' not in self.context:
            return []
        df = self.context['data']
        return sorted(df[df['Part'] == part]['Supplier'].unique())
    
    def analyze_part(self, part, supplier):
        if 'data' not in self.context:
            return "Please upload data first."
        
        df = self.context['data']
        part_df = df[(df['Part'] == part) & (df['Supplier'] == supplier)]
        
        if part_df.empty:
            return "No data available for the selected part and supplier."
        
        advantages = part_df.iloc[0]['Advantages']
        disadvantages = part_df.iloc[0]['Disadvantages']
        
        report = f"""
        **Part Analysis: {part} (Supplier: {supplier})**
        ---
        - **Advantages:** {advantages}
        - **Disadvantages:** {disadvantages}
        """
        return report

# Streamlit App
st.title("Automobile Part Analysis")

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = PartAnalysis()

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        result = st.session_state.analyzer.set_context(df)
        if result == "Data loaded successfully!":
            st.success(result)
            parts = st.session_state.analyzer.get_parts()
            if parts:
                selected_part = st.selectbox("Select a part to analyze:", parts)
                suppliers = st.session_state.analyzer.get_suppliers(selected_part)
                if suppliers:
                    selected_supplier = st.selectbox("Select a supplier:", suppliers)
                    if selected_supplier:
                        st.write(st.session_state.analyzer.analyze_part(selected_part, selected_supplier))
        else:
            st.error(result)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")


# # Configure page settings (must be the first Streamlit command)
# st.set_page_config(page_title="Product Analysis", page_icon="📝", layout="wide")

# # Apply common styling
# st.markdown(apply_common_style(), unsafe_allow_html=True)

# # Sidebar for language selection
# languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Urdu', 'Gujarati', 'Punjabi', 'Malayalam', 'Odia', 'Kannada', 'Assamese', 'Maithili', 'Sanskrit']
# selected_language = st.sidebar.selectbox("Select Language", languages)
# selected_lang_code = get_language_code(selected_language)
# # Wrap the title in the header div
# st.markdown(f'<div class="header"><h1>{translate_text("Product Analysis Dashboard", selected_lang_code)}</h1></div>', unsafe_allow_html=True)

# # Load SpaCy English model
# nlp = spacy.load("en_core_web_sm")

# class SentimentAnalysis:
#     def __init__(self):
#         self.context = {}

#     def set_context(self, df):
#         required_cols = ['Product Name', 'Supplier', 'Sentiment', 'Review Text']
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             return f"Missing required columns: {missing_cols}"
        
#         self.context['data'] = df.dropna(subset=['Sentiment']).reset_index(drop=True)
#         return "Data loaded successfully!"

#     def get_products(self):
#         if 'data' not in self.context:
#             return []
#         return sorted(self.context['data']['Product Name'].unique())

#     def transform_review(self, review):
#         """Converts first-person language to objective language using rule-based replacement."""
#         first_person_map = {
#             "I": "The user",
#             "me": "the user",
#             "my": "the user's",
#             "we": "customers",
#             "us": "customers",
#             "our": "customers'"
#         }
        
#         doc = nlp(review)
#         transformed_tokens = [first_person_map.get(token.text, token.text) for token in doc]
#         return ' '.join(transformed_tokens).strip()

#     def analyze_product(self, product):
#         if 'data' not in self.context:
#             return "Please upload data first."
        
#         df = self.context['data']
#         product_df = df[df['Product Name'] == product]
#         positive_reviews = product_df[product_df['Sentiment'] == 'Positive']['Review Text'].tolist()
#         negative_reviews = product_df[product_df['Sentiment'] == 'Negative']['Review Text'].tolist()
        
#         formatted_positive_reviews = ', '.join([self.transform_review(review) for review in positive_reviews[:3]]) if positive_reviews else 'No positive reviews available'
#         formatted_negative_reviews = ', '.join([self.transform_review(review) for review in negative_reviews[:3]]) if negative_reviews else 'No negative reviews available'
        
#         report = f"""
#         **Product Analysis: {product}**
#         ---
#         - **Advantages:** {formatted_positive_reviews}
#         - **Disadvantages:** {formatted_negative_reviews}
#         """
#         return report

# def main():
#     # Initialize session state
#     if 'analyzer' not in st.session_state:
#         st.session_state.analyzer = SentimentAnalysis()

#     # Main content section
#     st.markdown('<div class="content-section">', unsafe_allow_html=True)
#     uploaded_file = st.file_uploader(translate_text("Upload CSV file", selected_lang_code), type=['csv'])
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             result = st.session_state.analyzer.set_context(df)
#             if result == "Data loaded successfully!":
#                 st.success(result)
#                 products = st.session_state.analyzer.get_products()
#                 if products:
#                     selected_product = st.selectbox("Select a product to analyze:", products)
#                     if selected_product:
#                         st.write(st.session_state.analyzer.analyze_product(selected_product))
#             else:
#                 st.error(result)
#         except Exception as e:
#             st.error(f"Error processing file: {str(e)}")
#     st.markdown('</div>', unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
