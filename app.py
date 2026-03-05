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
    st.info("Comparativa entre segmentación por palabras (Word-level) y sub-palabras (BPE).")
    
    text_input = st.text_area("Texto a analizar:", "El procesamiento de lenguaje natural en EAFIT es fascinante.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Word-level (NLTK)")
        tokens_word = nltk.word_tokenize(text_input)
        st.write(tokens_word)
        st.metric("Total Tokens", len(tokens_word))

    with col2:
        st.subheader("BPE (Llama Style via Tiktoken)")
        # cl100k_base es el estándar para modelos modernos (similar a Llama 3)
        enc = tiktoken.get_encoding("cl100k_base")
        tokens_bpe_ids = enc.encode(text_input)
        tokens_bpe_decoded = [enc.decode([t]) for t in tokens_bpe_ids]
        st.write(tokens_bpe_decoded)
        st.metric("Total Tokens", len(tokens_bpe_ids))

# --- SECCIÓN 2: VECTORIZACIÓN ---
elif app_section == "Vectorización Clásica":
    st.header("📊 Vectorización: BoW y TF-IDF")
    corpus_input = st.text_area("Ingresa un corpus (una frase por línea):", 
                               "La minería de datos es técnica.\nEl aprendizaje profundo es potente.\nEAFIT lidera en datos.")
    
    corpus = [line.strip() for line in corpus_input.split('\n') if line.strip()]
    
    if corpus:
        tab1, tab2 = st.tabs(["Bolsa de Palabras (BoW)", "TF-IDF"])
        
        with tab1:
            cv = CountVectorizer()
            bow_res = cv.fit_transform(corpus)
            df_bow = pd.DataFrame(bow_res.toarray(), columns=cv.get_feature_names_out())
            st.dataframe(df_bow, use_container_width=True)

        with tab2:
            tfidf = TfidfVectorizer()
            tfidf_res = tfidf.fit_transform(corpus)
            df_tfidf = pd.DataFrame(tfidf_res.toarray(), columns=tfidf.get_feature_names_out())
            st.dataframe(df_tfidf, use_container_width=True)

# --- SECCIÓN 3: MODELADO DE SECUENCIAS ---
elif app_section == "Modelado de Secuencias":
    st.header("🔗 Análisis de Secuencias")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        seq_input = st.text_input("Texto para N-grams:", "Inteligencia artificial generativa")
        n_val = st.slider("Valor de N", 2, 3, 2)
        tokens = nltk.word_tokenize(seq_input.lower())
        gramas = list(ngrams(tokens, n_val))
        st.write(f"**{n_val}-gramas:**", gramas)
    
    st.divider()
    st.subheader("Cuadro Comparativo: RNN vs LSTM vs GRU")
    data_comp = {
        "Criterio": ["Memoria", "Arquitectura", "Entrenamiento", "Uso Ideal"],
        "RNN": ["Corto plazo (Desvanece)", "Capa oculta simple", "Rápido", "Secuencias muy cortas"],
        "LSTM": ["Largo plazo (Cell State)", "3 Compuertas (Input/Forget/Output)", "Lento (Complejo)", "Dependencias complejas"],
        "GRU": ["Largo plazo (Update Gate)", "2 Compuertas (Reset/Update)", "Moderado", "Eficiencia en memoria"]
    }
    st.table(pd.DataFrame(data_comp))

# --- SECCIÓN 4: LABORATORIO LLM ---
elif app_section == "Laboratorio LLM":
    st.header("🧪 Laboratorio de Hiperparámetros")
    
    col_p, col_r = st.columns([1, 2])
    with col_p:
        temp = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.1)
        st.caption("0.0 = Determinista | 2.0 = Creativo/Aleatorio")
        prompt_lab = st.text_area("Escribe un prompt:", "Define 'Deep Learning' en tono poético.")
        enviar = st.button("Generar Respuesta")

    if enviar:
        with col_r:
            with st.spinner("Consultando a Llama..."):
                resp = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt_lab}],
                    temperature=temp
                )
                st.markdown(f"### Resultado:\n{resp.choices[0].message.content}")

# --- SECCIÓN 5: AGENTE ESPECIALIZADO ---
elif app_section == "Agente Especializado":
    st.header("🤖 Agente: Consultor Técnico EAFIT")
    st.write("Agente experto que evalúa su propio desempeño en tiempo real.")
    
    query = st.chat_input("Pregunta al consultor sobre Ciencia de Datos...")
    
    if query:
        # 1. Ejecución del Agente
        t_inicio = time.time()
        respuesta_agente = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "Eres un consultor experto en Ciencia de Datos. Proporcionas respuestas técnicas, estructuradas y precisas."},
                {"role": "user", "content": query}
            ]
        )
        t_fin = time.time()
        
        # Procesamiento de métricas
        contenido = respuesta_agente.choices[0].message.content
        latencia = t_fin - t_inicio
        tokens_salida = respuesta_agente.usage.completion_tokens
        tps = tokens_salida / latencia if latencia > 0 else 0
        
        # 2. LLM-as-a-judge (Auto-evaluación)
        judge_prompt = f"""
        Actúa como un evaluador crítico. Califica del 1 al 10 la siguiente respuesta técnica basándote en la veracidad.
        Pregunta: {query}
        Respuesta: {contenido}
        Devuelve el formato: [Nota]/10 - [Breve explicación]
        """
        evaluacion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": judge_prompt}]
        )

        # Interfaz de resultados
        st.markdown("### 💬 Respuesta del Agente")
        st.info(contenido)
        
        st.divider()
        st.subheader("📊 Métricas de Desempeño (Real-time)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Latencia", f"{latencia:.2f} s")
        c2.metric("Tokens/Seg (TPS)", f"{tps:.2f}")
        c3.metric("Auto-Evaluación", evaluacion.choices[0].message.content.split(' - ')[0])
        
        st.write(f"**Justificación del Juez:** {evaluacion.choices[0].message.content}")
