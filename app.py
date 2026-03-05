import streamlit as st
import pandas as pd
import time
import tiktoken
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.util import ngrams
import plotly.express as px

# --- CORRECCIÓN DE ERROR NLTK ---
def inicializar_nltk():
    recursos = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']
    for recurso in recursos:
        try:
            nltk.data.find(f'tokenizers/{recurso}' if 'punkt' in recurso else f'taggers/{recurso}')
        except LookupError:
            nltk.download(recurso, quiet=True)

inicializar_nltk()
# --------------------------------

# Configuración de la página
st.set_page_config(page_title="Taller NLP & LLM - EAFIT", layout="wide")

# Inicialización de estado para la API Key
if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = ""

with st.sidebar:
    st.title("⚙️ Configuración")
    api_key_input = st.text_input("Introduce tu Groq API Key:", type="password")
    if api_key_input:
        st.session_state.GROQ_API_KEY = api_key_input
    
    selected_model = st.selectbox("Modelo Llama", ["llama-3.3-70b-versatile", "llama3-8b-8192"])
    
    st.divider()
    app_section = st.radio("Secciones del Taller", 
                          ["Tokenización & Encoding", "Vectorización Clásica", 
                           "Modelado de Secuencias", "Laboratorio LLM", "Agente Especializado"])

# Validación de API Key
if not st.session_state.GROQ_API_KEY:
    st.warning("⚠️ Por favor, introduce tu API Key en la barra lateral para comenzar.")
    st.stop()

client = Groq(api_key=st.session_state.GROQ_API_KEY)

# --- SECCIÓN 1: TOKENIZACIÓN ---
if app_section == "Tokenización & Encoding":
    st.header("🔤 Tokenización y Encoding")
    st.info("Comparativa entre segmentación por palabras (Word-level) y sub
