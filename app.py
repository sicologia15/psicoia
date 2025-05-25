
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import librosa
import numpy as np
import soundfile as sf
from fer import FER
import cv2
from PIL import Image
import tempfile
import torch

# --- Configuracion de la app ---
st.set_page_config(page_title="Análisis Emocional Psicológico", layout="wide")
st.title("App de Psicología: Detección de Emociones en Texto, Imagen y Audio")

# --- Modelo de emociones en texto ---
st.subheader("1. Análisis de emociones en texto")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

text_input = st.text_area("Escribe algo para analizar emociones:")
if text_input:
    with st.spinner("Analizando texto..."):
        text_emotions = emotion_classifier(text_input)[0]
        df_text = pd.DataFrame(text_emotions).sort_values("score", ascending=False)
        fig_text = px.bar(df_text, x="label", y="score", title="Emociones detectadas en texto")
        st.plotly_chart(fig_text)
        top_emotion = df_text.iloc[0]["label"]
        st.success(f"Emoción predominante: {top_emotion}")

# --- Análisis emocional de imagen ---
st.subheader("2. Análisis de emociones en imagen")
image_file = st.file_uploader("Sube una imagen de un rostro:", type=["jpg", "jpeg", "png"])
if image_file is not None:
    img = Image.open(image_file).convert('RGB')
    img_np = np.array(img)
    detector = FER(mtcnn=True)
    with st.spinner("Analizando imagen..."):
        result = detector.detect_emotions(img_np)
        if result:
            top_emotion = max(result[0]['emotions'], key=result[0]['emotions'].get)
            st.image(img, caption=f"Emoción detectada: {top_emotion}")
            df_img = pd.DataFrame.from_dict(result[0]['emotions'], orient='index', columns=['score']).reset_index()
            df_img.columns = ['emotion', 'score']
            fig_img = px.bar(df_img, x='emotion', y='score', title='Emociones faciales detectadas')
            st.plotly_chart(fig_img)
        else:
            st.warning("No se detectó ninguna cara.")

# --- Análisis emocional de audio ---
st.subheader("3. Análisis de emociones en audio")
audio_file = st.file_uploader("Sube un archivo de audio (wav):", type=["wav"])
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    with st.spinner("Analizando audio..."):
        y, sr = librosa.load(tmp_path)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rmse = np.mean(librosa.feature.rms(y=y))
        energy = np.mean(librosa.feature.rms(y=y)[0])
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        st.write(f"ZCR: {zcr:.4f}, Energía: {energy:.4f}, Tempo: {tempo:.2f} bpm")

        # Clasificación simple por regla (solo demostración)
        if energy > 0.04 and tempo > 100:
            st.success("Emoción estimada: Alegría o Euforia")
        elif energy < 0.02:
            st.info("Emoción estimada: Tristeza o Calma")
        else:
            st.warning("Emoción estimada: Neutra o Incierta")
