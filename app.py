import streamlit as st
import pandas as pd
import time
import tiktoken
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.util import ngrams
import plotly.express as px

# --- RECURSOS NLTK ---
@st.cache_resource
def load_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

load_nltk()

st.set_page_config(page_title="NLP & LLM Dashboard | EAFIT", layout="wide")

# --- API KEY ---
# Reemplaza con tu clave real para ejecución directa
API_KEY = "TU_API_KEY_AQUI" 

if "api_key" not in st.session_state:
    st.session_state.api_key = API_KEY

with st.sidebar:
    st.title("🛠️ Configuración")
    st.session_state.api_key = st.text_input("Groq API Key:", value=st.session_state.api_key, type="password")
    model_name = st.selectbox("Modelo Llama", ["llama-3.3-70b-versatile", "llama3-8b-8192"])
    
    st.divider()
    menu = st.radio("Secciones del Taller:", ["Fundamentos NLP", "Chat Multivariante & Desempeño"])

if not st.session_state.api_key or st.session_state.api_key == "TU_API_KEY_AQUI":
    st.warning("⚠️ Ingresa tu API Key en la barra lateral para activar el aplicativo.")
    st.stop()

client = Groq(api_key=st.session_state.api_key)

# --- SECCIÓN 1: FUNDAMENTOS (NLP CLÁSICO) ---
if menu == "Fundamentos NLP":
    st.header("🧠 Análisis de Texto y Vectorización")
    
    tab1, tab2 = st.tabs(["Tokenización & N-Grams", "Vectorización (TF-IDF)"])
    
    with tab1:
        txt = st.text_area("Texto de prueba:", "El análisis de estabilidad es fundamental en la ingeniería geotécnica.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Word-level vs BPE")
            st.write("**NLTK:**", nltk.word_tokenize(txt))
            enc = tiktoken.get_encoding("cl100k_base")
            st.write("**BPE (Llama Style):**", [enc.decode([t]) for t in enc.encode(txt)])
        with c2:
            st.subheader("N-Grams")
            n = st.slider("Selecciona N", 2, 3, 2)
            st.write(list(ngrams(nltk.word_tokenize(txt.lower()), n)))

    with tab2:
        corpus = st.text_area("Corpus (una frase por línea):", 
                               "Suelos residuales en Antioquia\nAnálisis de taludes y suelos\nIngeniería en EAFIT").split('\n')
        if st.button("Generar Matrices"):
            tfidf = TfidfVectorizer()
            mtx = tfidf.fit_transform([c for c in corpus if c.strip()])
            st.dataframe(pd.DataFrame(mtx.toarray(), columns=tfidf.get_feature_names_out()), use_container_width=True)

# --- SECCIÓN 2: CHAT MULTIVARIANTE (LLM) ---
else:
    st.header("🤖 Laboratorio de LLM: Comparativa de Parámetros")
    
    # Panel de entrada de parámetros
    with st.expander("🎛️ Configuración de Hiperparámetros (Entrada)", expanded=True):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            max_tk = st.slider("Max Tokens", 10, 2048, 512)
        with col_p2:
            top_p_val = st.slider("Top-P (Nucleus Sampling)", 0.0, 1.0, 0.9)
        with col_p3:
            sys_prompt = st.text_input("System Prompt:", "Eres un asistente técnico experto.")

    user_query = st.chat_input("Escribe tu consulta técnica...")

    if user_query:
        # Configuraciones para las 3 respuestas variando Temperatura
        configs = [
            {"label": "Determinista", "temp": 0.1},
            {"label": "Equilibrado", "temp": 0.7},
            {"label": "Creativo", "temp": 1.5}
        ]
        
        results_data = []
        cols = st.columns(3)

        for i, cfg in enumerate(configs):
            with cols[i]:
                st.subheader(cfg["label"])
                st.caption(f"Temp: {cfg['temp']} | Top-p: {top_p_val}")
                
                start_t = time.perf_counter()
                
                # Llamada a la API de Groq incorporando nuevos parámetros
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=cfg["temp"],
                    max_tokens=max_tk,
                    top_p=top_p_val
                )
                
                end_t = time.perf_counter()
                
                # Cálculo de métricas
                latencia = end_t - start_t
                tokens_out = response.usage.completion_tokens
                tps = tokens_out / latencia if latencia > 0 else 0
                
                # Mostrar resultado
                st.info(response.choices[0].message.content)
                
                # Mostrar métricas individuales
                m1, m2 = st.columns(2)
                m1.metric("Latencia", f"{latencia:.2f}s")
                m2.metric("TPS", f"{tps:.1f}")
                
                results_data.append({
                    "Config": cfg["label"],
                    "Latencia": latencia,
                    "TPS": tps,
                    "Tokens": tokens_out
                })

        # --- VISUALIZACIÓN DE DESEMPEÑO ---
        st.divider()
        st.subheader("📊 Análisis de Desempeño del Modelo")
        df_metrics = pd.DataFrame(results_data)
        
        g1, g2 = st.columns(2)
        with g1:
            fig1 = px.bar(df_metrics, x="Config", y="Latencia", color="Config",
                          title="Latencia Total (Segundos)", labels={"Latencia": "Segundos"})
            st.plotly_chart(fig1, use_container_width=True)
        with g2:
            fig2 = px.line(df_metrics, x="Config", y="TPS", markers=True,
                           title="Velocidad (Tokens por Segundo)")
            st.plotly_chart(fig2, use_container_width=True)

        # --- LLM-AS-A-JUDGE (AUTO-EVALUACIÓN) ---
        st.divider()
        st.subheader("⚖️ Módulo de Validación (LLM-as-a-judge)")
        with st.spinner("Evaluando calidad de las respuestas..."):
            # Evaluamos la segunda respuesta (la equilibrada)
            eval_query = f"Evalúa la calidad técnica de esta respuesta de 1 a 10 y justifica: {results_data[1]}"
            judge_res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": eval_query}]
            )
            st.success(f"**Dictamen del Juez:** {judge_res.choices[0].message.content}")
