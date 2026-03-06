import streamlit as st
import time
from groq import Groq

# Simulación de la conexión (asegúrate de tener configurado tu cliente Groq)
client = Groq(api_key=st.session_state.GROQ_API_KEY)

st.title("🤖 Chat-Conversacional Personalizable")

# --- BLOQUE DE CONFIGURACIÓN DE PROMPT Y PARÁMETROS ---
with st.expander("🛠️ Configuración del Agente y Parámetros", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        system_prompt = st.text_area(
            "Definir el System Prompt (Rol del Agente):",
            "Eres un experto en geotecnia y minería que responde de forma técnica y concisa."
        )
        model_choice = st.selectbox("Modelo Open Source", ["llama-3.3-70b-versatile", "llama3-8b-8192"])
    
    with col2:
        temp = st.slider("Temperatura (Creatividad)", 0.0, 2.0, 0.7)
        max_tokens = st.number_input("Máximo de Tokens", 100, 4096, 1000)

# --- ENTRADA DEL USUARIO ---
user_query = st.text_input("Ingresa tu consulta sobre el tema de interés:")

if user_query:
    st.subheader("🚀 Comparativa de Respuestas y Métricas")
    
    # Vamos a generar 3 respuestas variando ligeramente la temperatura internamente 
    # o simplemente mostrando la respuesta del modelo configurado.
    
    cols = st.columns(3)
    
    # Simulación de 3 variantes de respuesta (puedes variar la temperatura en cada una)
    temps_to_test = [0.2, 1.0, 1.8] # Baja, Media, Alta creatividad
    
    for i, t in enumerate(temps_to_test):
        with cols[i]:
            st.markdown(f"**Variante {i+1} (Temp: {t})**")
            
            start_time = time.perf_counter()
            
            completion = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=t,
                max_tokens=max_tokens
            )
            
            end_time = time.perf_counter()
            
            # Cálculo de Métricas
            latency = end_time - start_time
            out_tokens = completion.usage.completion_tokens
            tps = out_tokens / latency if latency > 0 else 0
            
            # Mostrar Respuesta
            st.info(completion.choices[0].message.content)
            
            # Mostrar Desempeño
            st.write(f"⏱️ **Latencia:** {latency:.2f}s")
            st.write(f"⚡ **TPS:** {tps:.2f}")
