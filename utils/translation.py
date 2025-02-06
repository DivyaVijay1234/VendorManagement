import streamlit as st
from googletrans import Translator

def get_language_code(language):
    """Convert language name to code."""
    language_codes = {
        'English': 'en',
        'Hindi': 'hi',
        'Bengali': 'bn',
        'Telugu': 'te',
        'Marathi': 'mr',
        'Tamil': 'ta',
        'Urdu': 'ur',
        'Gujarati': 'gu',
        'Punjabi': 'pa',
        'Malayalam': 'ml',
        'Odia': 'or',
        'Kannada': 'kn',
        'Assamese': 'as',
        'Maithili': 'mai',
        'Sanskrit': 'sa'
    }
    return language_codes.get(language, 'en')

@st.cache_data(ttl=3600)  # Cache translations for 1 hour
def translate_text(text, target_lang_code):
    """Translate text to target language."""
    if target_lang_code == 'en':
        return text
    
    try:
        translator = Translator()
        translation = translator.translate(text, dest=target_lang_code)
        return translation.text
    except Exception as e:
        print(f"Translation error for '{text}': {str(e)}")
        return text  # Return original text if translation fails