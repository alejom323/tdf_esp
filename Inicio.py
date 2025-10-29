import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
import os
import glob

# Configuración de página Streamlit
st.set_page_config(
    page_title="Demo TF-IDF en Español",
    page_icon="🔍",
    layout="wide"
)

# Estilos personalizados: fondo azul claro, tipografía Rubik
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Rubik&display=swap');

        body, .stApp {
            background-color: #e3f2fd;
            font-family: 'Rubik', sans-serif;
        }

        section[data-testid="stSidebar"] {
            background-color: #e3f2fd !important;
            color: black;
        }

        section[data-testid="stSidebar"] * {
            color: black !important;
            font-family: 'Rubik', sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            font-family: 'Rubik', sans-serif !important;
        }
    </style>
""", unsafe_allow_html=True)

# Buscar imagen con nombre 'imagen-robt1' y cualquier extensión válida
image_files = glob.glob("imagen-robt1.*")
if image_files:
    st.image(image_files[0], use_container_width=True)
else:
    st.warning("⚠️ No se encontró ninguna imagen llamada 'imagen-robt1'. Verifica que esté en el mismo directorio y tenga una extensión válida como .png, .jpg o .webp.")

# Título principal
st.title("🔍 Demo TF-IDF en Español")

# Documentos de ejemplo
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    
    if st.button("¿Dónde juegan el perro y el gato?", use_container_width=True):
        st.session_state.question = "¿Dónde juegan el perro y el gato?"
        st.rerun()
    
    if st.button("¿Qué hacen los niños en el parque?", use_container_width=True):
        st.session_state.question = "¿Qué hacen los niños en el parque?"
        st.rerun()
        
    if st.button("¿Cuándo cantan los pájaros?", use_container_width=True):
        st.session_state.question = "¿Cuándo cantan los pájaros?"
        st.rerun()
        
    if st.button("¿Dónde suena la música alta?", use_container_width=True):
        st.session_state.question = "¿Dónde suena la música alta?"
        st.rerun()
        
    if st.button("¿Qué animal maúlla durante la noche?", use_container_width=True):
        st.session_state.question = "¿Qué animal maúlla durante la noche?"
        st.rerun()

if 'question' in st.session_state:
    question = st.session_state.question

if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )
        
        X = vectorizer.fit_transform(documents)
        
        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        st.markdown("### 🎯 Respuesta")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            st.warning(f"**Respuesta (baja confianza):** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")
