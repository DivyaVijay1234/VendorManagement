def apply_common_style():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&display=swap');
        
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
            font-size: 2.5rem;
            font-weight: 600;
            color: #64ffda;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 3px;
            text-shadow: 0 0 10px rgba(100, 255, 218, 0.3);
        }
        
        /* Content styling */
        .content-section {
            background: rgba(2, 12, 27, 0.7);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #1e2d3d;
        }
        
        /* Override Streamlit styles */
        .stApp {
            background: linear-gradient(135deg, #0a192f 0%, #000000 100%);
        }
        
        .stButton button {
            background-color: #64ffda;
            color: #0a192f;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
        }
        
        .stSelectbox > div > div {
            background-color: rgba(2, 12, 27, 0.7);
            border: 1px solid #1e2d3d;
        }
        
        .stTextInput > div > div {
            background-color: rgba(2, 12, 27, 0.7);
            border: 1px solid #1e2d3d;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0a192f;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #64ffda;
            border-radius: 5px;
        }
    </style>
    """ 