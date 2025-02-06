import streamlit as st
import pandas as pd

# Configure page settings
st.set_page_config(
    page_title="Carbon Emission Estimator",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS to match main app styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&display=swap');
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0a192f 0%, #000000 100%);
        padding: 2rem;
    }
    
    /* Override Streamlit defaults */
    .stApp {
        background: linear-gradient(135deg, #0a192f 0%, #000000 100%);
    }
    
    .stMarkdown {
        color: #8892b0;
    }
    
    h1, h2, h3 {
        color: #64ffda !important;
        font-family: 'Orbitron', sans-serif;
    }
    
    /* Custom container for the upload and analysis section */
    .analysis-container {
        background: rgba(2, 12, 27, 0.7);
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #1e2d3d;
    }
    
    /* Style for the file uploader */
    .uploadedFile {
        border: 1px solid #64ffda;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Style for the select box */
    .stSelectbox label {
        color: #ccd6f6 !important;
    }
    
    /* Style for the results */
    .stMetric {
        background: rgba(100, 255, 218, 0.1);
        padding: 1rem;
        border-radius: 4px;
    }
    
    /* Style for text elements */
    p, label {
        color: #8892b0 !important;
    }
    
    strong {
        color: #ccd6f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Predefined emission factors and suggestions by material type
emission_factors = {
    "Plastic": {"factor": 1.5, "suggestion": "Use biodegradable plastics or reduce plastic usage altogether."},
    "Steel": {"factor": 2.0, "suggestion": "Opt for recycled steel or lightweight alternatives to reduce emissions."},
    "Rubber": {"factor": 1.8, "suggestion": "Source rubber from sustainable plantations or use synthetic alternatives."},
    "Aluminum": {"factor": 1.2, "suggestion": "Use recycled aluminium or explore lightweight alternatives."}
}

def main():
    # Title with custom styling
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Carbon Emission Estimator</h1>", unsafe_allow_html=True)
    
    # Create a container for the main content
    with st.container():
        st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
        
        # File upload section
        st.markdown("### Upload Component Data")
        st.write("Please upload a CSV file containing component and material information.")
        uploaded_file = st.file_uploader("Choose a file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                if "invoice_line_text" in df.columns and "material" in df.columns:
                    # Component selection
                    st.markdown("### Component Analysis")
                    unique_components = df["invoice_line_text"].unique()
                    selected_component = st.selectbox("Select a component to analyze:", unique_components)

                    # Analysis section
                    selected_row = df[df["invoice_line_text"] == selected_component]

                    if not selected_row.empty:
                        material = selected_row.iloc[0]["material"]

                        if material in emission_factors:
                            emissions = emission_factors[material]["factor"] * 10
                            suggestion = emission_factors[material]["suggestion"]

                            # Results display with metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Carbon Emissions", f"{emissions} kg CO₂")
                            with col2:
                                st.markdown(f"**Material Type:** {material}")
                            
                            # Suggestion box
                            st.markdown("### Reduction Suggestions")
                            st.info(suggestion)
                        else:
                            st.warning("Material not recognized for emission estimation.")
                    else:
                        st.error("Component not found in the data.")
                else:
                    st.error("The uploaded file must contain 'invoice_line_text' and 'material' columns.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()