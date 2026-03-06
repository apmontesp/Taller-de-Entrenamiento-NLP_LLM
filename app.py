import streamlit as st
import pandas as pd
import time
import tiktoken
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.util import ngrams

# --- CONFIGURACIÓN Y RECURSOS NLTK ---
def setup_nltk():
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

setup_nltk()

st.set_page_config(page_title="Taller NLP & LLM - EAFIT", layout="wide")

# --- MANEJO DE API KEY ---
# Reemplaza 'TU_API_KEY_AQUI' con tu llave real si quieres que sea automática
# O déjala vacía para ingresarla en la interfaz.
DEFAULT_API_KEY = "TU_API_KEY_AQUI" 

if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = DEFAULT_API_KEY

with st.sidebar:
    st.title("⚙️ Configuración")
    api_input = st.text_input("Groq API Key:", value=st.session_state.GROQ_API_KEY, type="password")
    st.session_state.GROQ_API_KEY = api_input
    
    selected_model = st.selectbox("Modelo", ["llama-3.3-70b-versatile", "llama3-8b-8192"])
    
    st.divider()
    section = st.radio("Menú del Taller:", 
                      ["Tokenización & Encoding", "Vectorización Clásica", 
                       "Modelado de Secuencias", "Chat Multivariante (Agente)"])

if not st.session_state.GROQ_API_KEY or st.session_state.GROQ_API_KEY == "TU_API_KEY_AQUI":
    st.warning("⚠️ Por favor, ingresa una API Key válida en la barra lateral.")
    st.stop()

client = Groq(api_key=st.session_state.GROQ_API_KEY)

# --- SECCIONES DE NLP CLÁSICO ---
if section == "Tokenización & Encoding":
    st.header("🔤 Tokenización y Encoding")
    text = st.text_area("Texto:", "El procesamiento de lenguaje natural en EAFIT.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Word-level (NLTK)")
        st.write(nltk.word_tokenize(text))
    with c2:
        st.subheader("BPE (Llama Style)")
        enc = tiktoken.get_encoding("cl100k_base")
        st.write([enc.decode([t]) for t in enc.encode(text)])

elif section == "Vectorización Clásica":
    st.header("📊 BoW y TF-IDF")
    corpus = st.text_area("Corpus (línea por frase):", "IA en EAFIT\nCiencia de datos\nIA y datos").split('\n')
    corpus = [c for c in corpus if c.strip()]
    if corpus:
        cv = CountVectorizer()
        tfidf = TfidfVectorizer()
        st.write("**Matriz BoW:**", pd.DataFrame(cv.fit_transform(corpus).toarray(), columns=cv.get_feature_names_out()))
        st.write("**Matriz TF-IDF:**", pd.DataFrame(tfidf.fit_transform(corpus).toarray(), columns=tfidf.get_feature_names_out()))

elif section == "Modelado de Secuencias":
    st.header("🔗 N-Grams y Arquitecturas")
    txt = st.text_input("Texto:", "La maestría en ciencia de datos")
    n = st.slider("N", 2, 3)
    st.write(f"{n}-grams:", list(ngrams(nltk.word_tokenize(txt.lower()), n)))
    
    st.table(pd.DataFrame({
        "Modelo": ["RNN", "LSTM", "GRU"],
        "Ventaja": ["Simplicidad", "Memoria larga", "Eficiencia"],
        "Desventaja": ["Gradiente desvaneciente", "Costosa", "Menos potente que LSTM"]
    }))

# --- SECCIÓN: AGENTE CON 3 RESPUESTAS Y MÉTRICAS ---
elif section == "Chat Multivariante (Agente)":
    st.header("🤖 Agente Especializado: 3 Respuestas Diferentes")
    
    system_prompt = st.text_area("Personalizar System Prompt:", 
                                "Eres un experto en Ciencia de Datos de EAFIT.")
    user_query = st.text_input("Tu pregunta:")
    
    if user_query:
        # Definimos 3 niveles de temperatura para ver el cambio de parámetros
        configuraciones = [
            {"name": "Determinista (Factual)", "temp": 0.1},
            {"name": "Equilibrado", "temp": 0.7},
            {"name": "Creativo (Variado)", "temp": 1.5}
        ]
        
        cols = st.columns(3)
        
        for i, config in enumerate(configuraciones):
            with cols[i]:
                st.subheader(config["name"])
                st.caption(f"Temperatura: {config['temp']}")
                
                t_start = time.perf_counter()
                
                # Llamada a la API
                res = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=config["temp"]
                )
                
                t_end = time.perf_counter()
                
                # Métricas
                latencia = t_end - t_start
                tokens = res.usage.completion_tokens
                tps = tokens / latencia if latencia > 0 else 0
                
                # Mostrar respuesta y métricas
                st.info(res.choices[0].message.content)
                
                st.metric("Latencia", f"{latencia:.2s} s")
                st.metric("TPS", f"{tps:.1f}")
                
                # Auto-evaluación (LLM-as-a-judge) para la primera respuesta
                if i == 1: # Evaluamos la equilibrada como muestra
                    judge = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": f"Califica del 1 al 10 esta respuesta: {res.choices[0].message.content}"}]
                    )
                    st.write(f"**Nota Juez:** {judge.choices[0].message.content[:20]}...")
