import streamlit as st
import streamlit.components.v1 as components

# Configure page settings
st.set_page_config(
    page_title="Leveraging LLM’S for AI-Driven Demand Prediction",
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
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1.5rem;
        padding: 1rem;
        max-width: 900px;
        margin: 0 auto;
        grid-template-rows: auto auto;  /* Explicitly define two rows */
    }
    
    /* Feature card styling */
    .feature-card {
        background: rgba(2, 12, 27, 0.7);
        border-radius: 8px;
        padding: 1.2rem;
        color: #8892b0;
        transition: all 0.3s ease;
        border: 1px solid #1e2d3d;
        height: 250px;  /* Fixed height for all cards */
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
    card_html = f"""
    <a href="{page_link}" target="_self" style="text-decoration: none;">
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <ul class="feature-list">
    """
    for feature in features:
        card_html += f"<li>• {feature}</li>"
    card_html += """
            </ul>
        </div>
    </a>
    """
    return card_html

def main():
    # Header
    st.markdown('<div class="header"><h1>AI-Driven Demand Prediction</h1></div>', unsafe_allow_html=True)
    
    # Feature cards container
    st.markdown('<div class="grid-container">', unsafe_allow_html=True)
    
    # Data Analysis Dashboard
    data_analysis_features = [
        "Data validation and preview",
        "Exploratory Data Analysis",
        "Time Series Analysis",
        "Demand Forecasting"
    ]
    st.markdown(create_feature_card("📊", "Data Analysis Dashboard", data_analysis_features, "Data_Analysis"), unsafe_allow_html=True)
    
    # AI Assistant
    ai_assistant_features = [
        "Get demand predictions",
        "Analyze specific parts",
        "Understand inventory trends",
        "Compare forecasting models"
    ]
    st.markdown(create_feature_card("🤖", "AI Assistant", ai_assistant_features, "AI_Assistant"), unsafe_allow_html=True)
    
    # Product Analysis
    product_analysis_features = [
        "Understand customer feedback",
        "Identify advantages and disadvantages",
        "Track product performance",
        "Generate insights"
    ]
    st.markdown(create_feature_card("📝", "Product Analysis", product_analysis_features, "Product_Analysis"), unsafe_allow_html=True)
    
    # Sentiment Analysis
    sentiment_analysis_features = [
        "Understand vendor performance",
        "Compare sentiment across suppliers",
        "Track satisfaction trends",
        "Identify improvement areas"
    ]
    st.markdown(create_feature_card("😊", "Sentiment Analysis", sentiment_analysis_features, "Sentiment_Analysis"), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()