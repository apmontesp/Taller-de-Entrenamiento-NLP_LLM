import streamlit as st
import pandas as pd
import time
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.util import ngrams
import plotly.express as px

# Configuración inicial
st.set_page_config(page_title="Taller NLP & LLM - EAFIT", layout="wide")
client = Groq(api_key="TU_API_KEY_AQUI")

# --- BARRA LATERAL ---
st.sidebar.title("Configuración")
app_mode = st.sidebar.selectbox("Selecciona una sección", 
    ["Tokenización & BoW", "Modelado de Secuencias", "Laboratorio LLM", "Agente Especializado"])

# --- SECCIÓN 1: TOKENIZACIÓN Y VECTORIZACIÓN ---
if app_mode == "Tokenización & BoW":
    st.header("1. Tokenización y Vectorización Clásica")
    user_text = st.text_area("Ingresa un corpus (separa frases por punto):", 
                             "El procesamiento de lenguaje es fascinante. Los LLMs son potentes.")
    
    col1, col2 = st.columns(2)
    
    if user_text:
        # BoW y TF-IDF
        corpus = user_text.split('.')
        corpus = [t.strip() for t in corpus if t.strip() != ""]
        
        vectorizer_bow = CountVectorizer()
        bow_matrix = vectorizer_bow.fit_transform(corpus)
        
        vectorizer_tfidf = TfidfVectorizer()
        tfidf_matrix = vectorizer_tfidf.fit_transform(corpus)
        
        with col1:
            st.subheader("Matriz BoW")
            st.write(pd.DataFrame(bow_matrix.toarray(), columns=vectorizer_bow.get_feature_names_out()))
            
        with col2:
            st.subheader("Valores TF-IDF")
            st.write(pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer_tfidf.get_feature_names_out()))

# --- SECCIÓN 2: MODELADO DE SECUENCIAS ---
elif app_mode == "Modelado de Secuencias":
    st.header("2. N-Grams y Evolución de Arquitecturas")
    text_input = st.text_input("Texto para N-Grams:", "Inteligencia artificial generativa en EAFIT")
    
    tokens = text_input.split()
    bi_grams = list(ngrams(tokens, 2))
    tri_grams = list(ngrams(tokens, 3))
    
    st.write(f"**Bi-gramas:** {bi_grams}")
    st.write(f"**Tri-gramas:** {tri_grams}")
    
    st.divider()
    st.subheader("Comparativo: RNN vs LSTM vs GRU")
    data = {
        "Arquitectura": ["RNN", "LSTM", "GRU"],
        "Eficiencia": ["Baja (Desvanecimiento de gradiente)", "Alta (Gating mechanisms)", "Media/Alta (Menos parámetros que LSTM)"],
        "Memoria": ["Corto plazo", "Largo plazo (Cell State)", "Largo plazo (Update/Reset gates)"]
    }
    st.table(pd.DataFrame(data))

# --- SECCIÓN 3: LABORATORIO LLM & AGENTE ---
elif app_mode == "Laboratorio LLM":
    st.header("3. Playground de Creatividad (Temperatura)")
    temp = st.slider("Temperatura", 0.0, 2.0, 0.7, step=0.1)
    prompt = st.text_input("Escribe un inicio de historia:")
    
    if st.button("Generar"):
        start_time = time.time()
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            stream=False
        )
        end_time = time.time()
        
        respuesta = completion.choices[0].message.content
        st.write(f"**Respuesta:** {respuesta}")
        
        # Métricas de Groq
        st.info(f"""
        ⏱️ Latencia: {round(end_time - start_time, 2)}s | 
        ⚡ TPS: {completion.usage.completion_tokens / (end_time - start_time):.2f} | 
        Tokens: {completion.usage.total_tokens}
        """)

elif app_mode == "Agente Especializado":
    st.header("4. Agente: Consultor Académico")
    # Aquí implementarías la lógica de LLM-as-a-judge
    st.write("Módulo para evaluar calidad de 1 a 10 usando un segundo llamado a la API.")
