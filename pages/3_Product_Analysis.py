import streamlit as st
import pandas as pd
import spacy
from utils.style import apply_common_style
from utils.translation import translate_text, get_language_code

# Configure page settings
st.set_page_config(page_title="Product Analysis", page_icon="📝", layout="wide")

# Apply common styling
st.markdown(apply_common_style(), unsafe_allow_html=True)

# Sidebar for language selection
languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Urdu', 'Gujarati', 'Punjabi', 'Malayalam', 'Odia', 'Kannada', 'Assamese', 'Maithili', 'Sanskrit']
selected_language = st.sidebar.selectbox("Select Language", languages)
selected_lang_code = get_language_code(selected_language)

# Wrap the title in the header div
st.markdown(f'<div class="header"><h1>{translate_text("Product Analysis Dashboard", selected_lang_code)}</h1></div>', unsafe_allow_html=True)

class PartAnalysis:
    def __init__(self):
        self.context = {}

    def set_context(self, df):
        required_cols = ['Part', 'Supplier', 'Advantages', 'Disadvantages']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return translate_text(f"Missing required columns: {missing_cols}", selected_lang_code)
        
        self.context['data'] = df.dropna().reset_index(drop=True)
        return translate_text("Data loaded successfully!", selected_lang_code)

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
            return translate_text("Please upload data first.", selected_lang_code)
        
        df = self.context['data']
        part_df = df[(df['Part'] == part) & (df['Supplier'] == supplier)]
        
        if part_df.empty:
            return translate_text("No data available for the selected part and supplier.", selected_lang_code)
        
        advantages = part_df.iloc[0]['Advantages']
        disadvantages = part_df.iloc[0]['Disadvantages']
        
        report = f"""
        **{translate_text('Part Analysis', selected_lang_code)}: {part} ({translate_text('Supplier', selected_lang_code)}: {supplier})**
        ---
        - **{translate_text('Advantages', selected_lang_code)}:** {advantages}
        - **{translate_text('Disadvantages', selected_lang_code)}:** {disadvantages}
        """
        return report

def main():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PartAnalysis()

    uploaded_file = st.file_uploader(translate_text("Upload CSV file", selected_lang_code), type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            result = st.session_state.analyzer.set_context(df)
            if result == translate_text("Data loaded successfully!", selected_lang_code):
                st.success(result)
                parts = st.session_state.analyzer.get_parts()
                if parts:
                    selected_part = st.selectbox(translate_text("Select a part to analyze:", selected_lang_code), parts)
                    suppliers = st.session_state.analyzer.get_suppliers(selected_part)
                    if suppliers:
                        selected_supplier = st.selectbox(translate_text("Select a supplier:", selected_lang_code), suppliers)
                        if selected_supplier:
                            st.write(st.session_state.analyzer.analyze_part(selected_part, selected_supplier))
            else:
                st.error(result)
        except Exception as e:
            st.error(translate_text(f"Error processing file: {str(e)}", selected_lang_code))

if __name__ == "__main__":
    main()
