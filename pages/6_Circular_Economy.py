import streamlit as st
import pandas as pd
import plotly.express as px
from utils.style import apply_common_style
from utils.translation import translate_text, get_language_code
from sustainable_data import sustainability_info

# Configure page settings
st.set_page_config(page_title="Circular Economy Analysis", page_icon="♻️", layout="wide")

# Apply common styling
st.markdown(apply_common_style(), unsafe_allow_html=True)

# Sidebar for language selection
languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Urdu', 'Gujarati', 'Punjabi', 'Malayalam', 'Odia', 'Kannada', 'Assamese', 'Maithili', 'Sanskrit']
selected_language = st.sidebar.selectbox("Select Language", languages)
selected_lang_code = get_language_code(selected_language)

def load_and_validate_data(df):
    """Load and validate data with flexible column requirements."""
    minimum_required_cols = ['job_card_date', 'invoice_line_text', 'material']
    missing_cols = [col for col in minimum_required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(translate_text(f"Missing essential columns: {missing_cols}", selected_lang_code))
        return None
        
    try:
        data = df.copy()
        return data
    except Exception as e:
        st.error(translate_text(f"Error processing data: {str(e)}", selected_lang_code))
        return None

def display_material_distribution(data):
    """Display material distribution analysis."""
    material_counts = data['material'].value_counts()
    
    fig = px.pie(
        values=material_counts.values,
        names=material_counts.index,
        title=translate_text("Distribution of Materials in Spare Parts", selected_lang_code)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(translate_text("""
    This chart shows the distribution of different materials used in spare parts.
    Understanding this distribution helps in planning recycling and sustainability initiatives.
    """, selected_lang_code))

def display_sustainability_metrics(data, selected_part, material):
    """Display sustainability information and metrics."""
    info = sustainability_info.get(material, {
        "lifespan": "Unknown",
        "recycling": "No information available",
        "reuse": "No information available"
    })
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            translate_text("Estimated Lifespan", selected_lang_code),
            info['lifespan']
        )
    
    with col2:
        part_count = len(data[data['invoice_line_text'] == selected_part])
        st.metric(
            translate_text("Total Usage Count", selected_lang_code),
            part_count
        )
    
    with col3:
        material_count = len(data[data['material'] == material])
        st.metric(
            translate_text("Material Usage Count", selected_lang_code),
            material_count
        )

def main():
    st.markdown(f'<div class="header"><h1>{translate_text("Circular Economy Analysis", selected_lang_code)}</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    st.write(translate_text("""
    ## Business Case: Sustainable Parts Management
    This analysis helps understand the environmental impact of spare parts through their lifecycle,
    focusing on material sustainability, recycling potential, and reuse opportunities.
    """, selected_lang_code))
    
    uploaded_file = st.file_uploader(translate_text("Upload CSV file", selected_lang_code), type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        data = load_and_validate_data(df)
        
        if data is not None:
            # Material Distribution Analysis
            with st.expander(translate_text("Material Distribution Analysis", selected_lang_code)):
                display_material_distribution(data)
            
            # Part Selection and Analysis
            st.subheader(translate_text("Part-Specific Sustainability Analysis", selected_lang_code))
            spare_parts = data["invoice_line_text"].unique()
            selected_part = st.selectbox(
                translate_text("Select a spare part", selected_lang_code),
                spare_parts
            )
            
            material = data[data["invoice_line_text"] == selected_part]["material"].values[0]
            
            # Display sustainability metrics
            display_sustainability_metrics(data, selected_part, material)
            
            # Detailed Sustainability Information
            with st.expander(translate_text("Detailed Sustainability Information", selected_lang_code)):
                info = sustainability_info.get(material, {
                    "lifespan": "Unknown",
                    "recycling": "No information available",
                    "reuse": "No information available"
                })
                
                st.write(f"**{translate_text('Material', selected_lang_code)}:** {material}")
                st.write(f"**{translate_text('Recycling Methods', selected_lang_code)}:**")
                st.write(info['recycling'])
                st.write(f"**{translate_text('Reuse Options', selected_lang_code)}:**")
                st.write(info['reuse'])
            
            # Historical Usage Trends
            with st.expander(translate_text("Historical Usage Trends", selected_lang_code)):
                part_usage = data[data['invoice_line_text'] == selected_part].copy()
                part_usage['job_card_date'] = pd.to_datetime(part_usage['job_card_date'])
                monthly_usage = part_usage.groupby(pd.Grouper(key='job_card_date', freq='M')).size().reset_index()
                monthly_usage.columns = ['date', 'count']
                
                fig = px.line(
                    monthly_usage,
                    x='date',
                    y='count',
                    title=translate_text(f"Monthly Usage Trend for {selected_part}", selected_lang_code)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 