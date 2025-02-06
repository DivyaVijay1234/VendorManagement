import streamlit as st
import streamlit.components.v1 as components
from utils.translation import translate_text, get_language_code

# Configure page settings
st.set_page_config(
    page_title="Leveraging LLM'S for AI-Driven Demand Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&display=swap');
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0a192f 0%, #000000 100%);
        padding: 2rem;
        min-height: 100vh;
    }
    
    /* Header styling */
    .header {
        color: #ffffff;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: transparent;
    }
    
    .header h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 600;
        color: #64ffda;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
    }
    
    .header p {
        color: #8892b0;
        font-size: 1.1rem;
    }
    
    /* Grid container */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));  /* Changed to 3 columns */
        gap: 1.5rem;
        padding: 1rem;
        max-width: 1200px;  /* Increased max-width to accommodate 3 columns */
        margin: 0 auto;
    }
    
    /* Feature card styling */
    .feature-card {
        background: rgba(2, 12, 27, 0.7);
        border-radius: 8px;
        padding: 1.2rem;
        color: #8892b0;
        transition: all 0.3s ease;
        border: 1px solid #1e2d3d;
        height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #64ffda;
        box-shadow: 0 4px 20px rgba(100, 255, 218, 0.1);
    }
    
    .feature-icon {
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
        color: #64ffda;
    }
    
    .feature-title {
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
        color: #ccd6f6;
        font-weight: 600;
    }
    
    .feature-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    .feature-list li {
        margin-bottom: 0.4rem;
        color: #8892b0;
        font-size: 0.85rem;
        position: relative;
        padding-left: 1rem;
    }
    
    .feature-list li:before {
        content: "▹";
        position: absolute;
        left: 0;
        color: #64ffda;
    }

    /* Override Streamlit styles */
    .stApp {
        background: linear-gradient(135deg, #0a192f 0%, #000000 100%);
    }
</style>
""", unsafe_allow_html=True)

def create_feature_card(icon, title, features, page_link):
    # Update the page link format to match Streamlit's convention
    formatted_page_link = f"/{page_link}" if page_link != "Home" else "/"
    # Ensure features are properly escaped for HTML
    feature_items = "".join([f"<li>• {feature}</li>" for feature in features])
    
    card_html = f"""
    <a href="{formatted_page_link}" target="_self" style="text-decoration: none;">
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <ul class="feature-list">
                {feature_items}
            </ul>
        </div>
    </a>
    """
    return card_html

def main():
    # Language selection in sidebar
    languages = ['English', 'Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Urdu', 'Gujarati', 'Punjabi', 'Malayalam', 'Odia', 'Kannada', 'Assamese', 'Maithili', 'Sanskrit']
    
    # Initialize with English
    selected_lang_code = 'en'
    
    # Now use it in the selectbox
    selected_language = st.sidebar.selectbox("Select Language", languages)
    selected_lang_code = get_language_code(selected_language)

    # Header with translation
    st.markdown(f'<div class="header"><h1>{translate_text("AI-Driven Inventory Demand Prediction", selected_lang_code)}</h1></div>', unsafe_allow_html=True)
    
    # Feature cards container
    st.markdown('<div class="grid-container">', unsafe_allow_html=True)
    
    # Data Analysis Dashboard with translation
    data_analysis_features = [
        translate_text("Data validation and preview", selected_lang_code),
        translate_text("Exploratory Data Analysis", selected_lang_code),
        translate_text("Time Series Analysis", selected_lang_code),
        translate_text("Demand Forecasting", selected_lang_code)
    ]
    
    # AI Assistant features with translation
    ai_assistant_features = [
        translate_text("Get demand predictions", selected_lang_code),
        translate_text("Analyze specific parts", selected_lang_code),
        translate_text("Understand inventory trends", selected_lang_code),
        translate_text("Compare forecasting models", selected_lang_code)
    ]
    
    # Product Analysis features with translation
    product_analysis_features = [
        translate_text("Understand customer feedback", selected_lang_code),
        translate_text("Identify advantages and disadvantages", selected_lang_code),
        translate_text("Track product performance", selected_lang_code),
        translate_text("Generate insights", selected_lang_code)
    ]
    
    # Sentiment Analysis features with translation
    sentiment_analysis_features = [
        translate_text("Understand vendor performance", selected_lang_code),
        translate_text("Compare sentiment across suppliers", selected_lang_code),
        translate_text("Track satisfaction trends", selected_lang_code),
        translate_text("Identify improvement areas", selected_lang_code)
    ]

    # Create feature cards with translated titles
    st.markdown(create_feature_card("📊", 
        translate_text("Data Analysis Dashboard", selected_lang_code), 
        data_analysis_features, 
        "Data_Analysis"), 
        unsafe_allow_html=True)
    
    st.markdown(create_feature_card("🤖", 
        translate_text("AI Assistant", selected_lang_code), 
        ai_assistant_features, 
        "AI_Assistant"), 
        unsafe_allow_html=True)
    
    st.markdown(create_feature_card("📝", 
        translate_text("Product Analysis", selected_lang_code), 
        product_analysis_features, 
        "Product_Analysis"), 
        unsafe_allow_html=True)
    
    st.markdown(create_feature_card("😊", 
        translate_text("Sentiment Analysis", selected_lang_code), 
        sentiment_analysis_features, 
        "Sentiment_Analysis"), 
        unsafe_allow_html=True)
    
    # Update the grid container features
    carbon_circular_features = [
        translate_text("Material sustainability analysis", selected_lang_code),
        translate_text("Carbon emission calculation", selected_lang_code),
        translate_text("Recycling potential assessment", selected_lang_code),
        translate_text("Environmental impact tracking", selected_lang_code),
        translate_text("Emission reduction suggestions", selected_lang_code)
    ]
    
    # Replace the separate Carbon Emission card with the combined version
    st.markdown(create_feature_card("♻️", 
        translate_text("Circular Economy & Emissions", selected_lang_code), 
        carbon_circular_features, 
        "Circular_Economy"), 
        unsafe_allow_html=True)
    
    # Vendor Demand Forecast
    vendor_demand_forecast_features = [
        translate_text("Upload CSV file", selected_lang_code),
        translate_text("Select Vendor", selected_lang_code),
        translate_text("Select Spare Part", selected_lang_code),
        translate_text("View Forecast", selected_lang_code),
        translate_text("View Positive Review Percentage", selected_lang_code)
    ]
    st.markdown(create_feature_card("📈", 
        translate_text("Sentiment-Based Demand Forecast", selected_lang_code), 
        vendor_demand_forecast_features, 
        "Vendor_Demand_Forecast"), 
        unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()