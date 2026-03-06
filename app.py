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

# --- API KEY & CLIENTE ---
# Puedes poner tu API Key aquí directamente
API_KEY = "TU_API_KEY_AQUI" 

if "api_key" not in st.session_state:
    st.session_state.api_key = API_KEY

with st.sidebar:
    st.title("🛠️ Configuración")
    st.session_state.api_key = st.text_input("Groq API Key:", value=st.session_state.api_key, type="password")
    model_name = st.selectbox("Modelo", ["llama-3.3-70b-versatile", "llama3-8b-8192"])
    
    st.divider()
    menu = st.radio("Secciones:", ["Fundamentos NLP", "Agente & Comparativa"])

if not st.session_state.api_key or st.session_state.api_key == "TU_API_KEY_AQUI":
    st.warning("⚠️ Ingresa tu API Key en la barra lateral.")
    st.stop()

client = Groq(api_key=st.session_state.api_key)

# --- SECCIÓN 1: FUNDAMENTOS ---
if menu == "Fundamentos NLP":
    st.header("🧠 Fundamentos de NLP")
    
    tab1, tab2, tab3 = st.tabs(["Tokenización", "Vectorización", "Secuencias"])
    
    with tab1:
        txt = st.text_input("Texto para tokenizar:", "El procesamiento de lenguaje natural es clave.")
        c1, c2 = st.columns(2)
        c1.write("**Word-level (NLTK):**")
        c1.write(nltk.word_tokenize(txt))
        c2.write("**BPE (Llama-style):**")
        enc = tiktoken.get_encoding("cl100k_base")
        c2.write([enc.decode([t]) for t in enc.encode(txt)])

    with tab2:
        corpus = st.text_area("Corpus (línea por frase):", "Análisis de suelos\nEstabilidad de taludes\nAnálisis de taludes").split('\n')
        if st.button("Vectorizar"):
            tfidf = TfidfVectorizer()
            mtx = tfidf.fit_transform([c for c in corpus if c.strip()])
            df = pd.DataFrame(mtx.toarray(), columns=tfidf.get_feature_names_out())
            st.dataframe(df)

    with tab3:
        n_txt = st.text_input("N-Grams:", "Inteligencia artificial generativa")
        n = st.slider("N", 2, 3)
        st.write(list(ngrams(nltk.word_tokenize(n_txt.lower()), n)))

# --- SECCIÓN 2: AGENTE CON 3 RESPUESTAS Y GRÁFICOS ---
else:
    st.header("🤖 Agente Multivariante & Desempeño")
    
    col_cfg, col_sys = st.columns([1, 2])
    with col_cfg:
        temp_input = st.slider("Temperatura base", 0.0, 2.0, 0.7)
    with col_sys:
        sys_prompt = st.text_area("System Prompt (Personalidad):", 
                                  "Eres un consultor experto de la Universidad EAFIT.")

    user_query = st.chat_input("Escribe tu consulta aquí...")

    if user_query:
        # Configuraciones para las 3 respuestas
        configs = [
            {"label": "Conservador", "temp": 0.2},
            {"label": "Equilibrado", "temp": 0.8},
            {"label": "Creativo", "temp": 1.6}
        ]
        
        results = []
        cols = st.columns(3)

        for i, cfg in enumerate(configs):
            with cols[i]:
                st.subheader(cfg["label"])
                st.caption(f"Temp: {cfg['temp']}")
                
                t0 = time.perf_counter()
                
                # Llamada a Groq
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=cfg["temp"]
                )
                
                t1 = time.perf_counter()
                
                # Métricas
                latencia = t1 - t0
                tokens = resp.usage.completion_tokens
                tps = tokens / latencia if latencia > 0 else 0
                
                # Guardar para el gráfico
                results.append({
                    "Configuración": cfg["label"],
                    "Latencia (s)": latencia,
                    "TPS": tps,
                    "Tokens": tokens
                })
                
                st.info(resp.choices[0].message.content)
                st.metric("Latencia", f"{latencia:.2f} s") # CORRECCIÓN AQUÍ (.2f)
                st.metric("TPS", f"{tps:.1f}")

        # --- SECCIÓN DE GRÁFICOS ---
        st.divider()
        st.subheader("📈 Análisis Comparativo de Desempeño")
        
        df_res = pd.DataFrame(results)
        
        c_gra1, c_gra2 = st.columns(2)
        
        with c_gra1:
            fig_lat = px.bar(df_res, x="Configuración", y="Latencia (s)", 
                             title="Latencia por Configuración", color="Configuración")
            st.plotly_chart(fig_lat, use_container_width=True)
            
        with c_gra2:
            fig_tps = px.line(df_res, x="Configuración", y="TPS", 
                              title="Velocidad de Generación (TPS)", markers=True)
            st.plotly_chart(fig_tps, use_container_width=True)

        # Módulo LLM-as-a-judge (Auto-evaluación)
        st.divider()
        st.subheader("⚖️ Evaluación 'LLM-as-a-judge'")
        with st.spinner("El juez está evaluando la respuesta equilibrada..."):
            judge = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"Califica del 1 al 10 la veracidad de esta respuesta: {results[1]}"}]
            )
            st.success(f"Dictamen del Juez: {judge.choices[0].message.content}")
