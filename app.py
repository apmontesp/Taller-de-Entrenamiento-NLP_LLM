import streamlit as st
import pandas as pd
import time
import tiktoken
from groq import Groq
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.util import ngrams
import plotly.express as px

# Descarga de recursos necesarios para NLTK
nltk.download('punkt')

# Configuración de la página
st.set_page_config(page_title="Taller NLP & LLM - EAFIT", layout="wide")

# Inicialización de Groq (Asegúrate de configurar tu API Key en los secretos de Streamlit o localmente)
if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = ""

with st.sidebar:
    st.title("⚙️ Configuración")
    api_key = st.text_input("Introduce tu Groq API Key:", type="password")
    if api_key:
        st.session_state.GROQ_API_KEY = api_key
    
    selected_model = st.selectbox("Modelo Llama", ["llama-3.3-70b-versatile", "llama3-8b-8192"])
    app_section = st.radio("Secciones del Taller", 
                          ["Tokenización & Encoding", "Vectorización Clásica", 
                           "Modelado de Secuencias", "Laboratorio LLM", "Agente Especializado"])

if not st.session_state.GROQ_API_KEY:
    st.warning("Por favor, introduce tu API Key en la barra lateral para comenzar.")
    st.stop()

client = Groq(api_key=st.session_state.GROQ_API_KEY)

# --- SECCIÓN 1: TOKENIZACIÓN ---
if app_section == "Tokenización & Encoding":
    st.header("🔤 Tokenización y Encoding")
    text_input = st.text_area("Texto a fragmentar:", "El procesamiento de lenguaje natural en EAFIT es increíble.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Word-level Tokenization")
        tokens_word = nltk.word_tokenize(text_input)
        st.write(tokens_word)
        st.info(f"Total tokens (Word): {len(tokens_word)}")

    with col2:
        st.subheader("BPE Tokenization (Llama style)")
        # Usamos el encoding de cl100k_base que es similar al usado por modelos modernos
        enc = tiktoken.get_encoding("cl100k_base")
        tokens_bpe = enc.encode(text_input)
        tokens_bpe_decoded = [enc.decode([t]) for t in tokens_bpe]
        st.write(tokens_bpe_decoded)
        st.info(f"Total tokens (BPE): {len(tokens_bpe)}")

# --- SECCIÓN 2: VECTORIZACIÓN ---
elif app_section == "Vectorización Clásica":
    st.header("📊 Vectorización: BoW y TF-IDF")
    corpus_input = st.text_area("Ingresa varias frases (una por línea):", 
                               "El aprendizaje es clave.\nLa ciencia de datos es el futuro.\nEAFIT enseña ciencia de datos.")
    
    corpus = [line.strip() for line in corpus_input.split('\n') if line.strip()]
    
    if corpus:
        # BoW
        cv = CountVectorizer()
        bow_res = cv.fit_transform(corpus)
        df_bow = pd.DataFrame(bow_res.toarray(), columns=cv.get_feature_names_out())
        st.subheader("Matriz Bag of Words (BoW)")
        st.dataframe(df_bow)

        # TF-IDF
        tfidf = TfidfVectorizer()
        tfidf_res = tfidf.fit_transform(corpus)
        df_tfidf = pd.DataFrame(tfidf_res.toarray(), columns=tfidf.get_feature_names_out())
        st.subheader("Valores TF-IDF")
        st.dataframe(df_tfidf)

# --- SECCIÓN 3: MODELADO DE SECUENCIAS ---
elif app_section == "Modelado de Secuencias":
    st.header("🔗 Análisis de Secuencias")
    seq_input = st.text_input("Texto para N-grams:", "La inteligencia artificial transforma el mundo")
    n_choice = st.slider("Selecciona N", 2, 3, 2)
    
    tokens = nltk.word_tokenize(seq_input.lower())
    n_grams = list(ngrams(tokens, n_choice))
    st.write(f"**{n_choice}-grams generados:**", n_grams)

    st.divider()
    st.subheader("Cuadro Comparativo: RNN vs LSTM vs GRU")
    comparison_data = {
        "Característica": ["Dependencias largas", "Arquitectura", "Eficiencia Computacional", "Problema del Gradiente"],
        "RNN": ["Pobre", "Simple (Capa oculta única)", "Alta (Menos parámetros)", "Sufre de desvanecimiento"],
        "LSTM": ["Excelente", "Compleja (3 compuertas + Cell State)", "Baja (Más parámetros)", "Resuelve con Cell State"],
        "GRU": ["Muy buena", "Simplificada (2 compuertas)", "Media (Balanceada)", "Resuelve con Update Gate"]
    }
    st.table(pd.DataFrame(comparison_data))

# --- SECCIÓN 4: LABORATORIO LLM ---
elif app_section == "Laboratorio LLM":
    st.header("🧪 Laboratorio de LLM")
    temp = st.slider("Temperatura (Creatividad vs Coherencia)", 0.0, 2.0, 0.7, 0.1)
    prompt_lab = st.text_area("Mensaje:", "Explica qué es un vector en tres frases creativas.")
    
    if st.button("Probar"):
        with st.spinner("Generando..."):
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt_lab}],
                temperature=temp
            )
            st.markdown(f"**Respuesta:** {response.choices[0].message.content}")
            st.caption(f"Configuración aplicada: Temperatura {temp}")

# --- SECCIÓN 5: AGENTE Y MÉTRICAS ---
elif app_section == "Agente Especializado":
    st.header("🤖 Agente: Consultor de Ciencia de Datos EAFIT")
    
    user_query = st.chat_input("Hazle una pregunta técnica a tu agente...")
    
    if user_query:
        # 1. Ejecución del Agente
        start_time = time.time()
        agent_response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "Eres un experto Consultor Académico de la Maestría en Ciencia de Datos de EAFIT. Responde de forma técnica y precisa."},
                {"role": "user", "content": user_query}
            ]
        )
        end_time = time.time()
        
        # Datos de respuesta
        content = agent_response.choices[0].message.content
        latency = end_time - start_time
        usage = agent_response.usage
        tps = usage.completion_tokens / latency if latency > 0 else 0
        
        # 2. LLM-as-a-judge (Auto-evaluación)
        judge_prompt = f"""
        Actúa como un profesor de Maestría. Evalúa la siguiente respuesta del 1 al 10 basándote en la veracidad y precisión técnica.
        Pregunta del usuario: {user_query}
        Respuesta del agente: {content}
        Proporciona el puntaje numérico seguido de una breve justificación de una línea.
        """
        judge_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": judge_prompt}]
        )
        
        # Mostrar resultados
        st.markdown("### Respuesta del Agente")
        st.write(content)
        
        st.divider()
        st.subheader("📊 Métricas de Desempeño")
        m1, m2, m3 = st.columns(3)
        m1.metric("Latencia", f"{latency:.2f} s")
        m2.metric("Tokens por Segundo (TPS)", f"{tps:.2f}")
        m3.metric("Evaluación (Judge)", judge_response.choices[0].message.content.split()[0])
        
        with st.expander("Ver Justificación del Juez"):
            st.write(judge_response.choices[0].message.content)
