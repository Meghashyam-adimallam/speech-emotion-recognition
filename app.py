# app.py

import os
import sys
import streamlit as st
import numpy as np
import torch
import librosa

# Add 'src' folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import EmotionBiLSTM
from src.preprocessing import extract_mfcc

# Emotion labels
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionBiLSTM(input_dim=40, hidden_dim=128, num_layers=2, num_classes=8)
model.load_state_dict(torch.load("emotion_bilstm.pth", map_location=device))
model.to(device)
model.eval()

# Streamlit UI
st.set_page_config(page_title="🎙️ Emotion from Voice", layout="centered")
st.title("🎙️ Speech Emotion Recognition App")
st.write("Upload a `.wav` file and let the AI guess the emotion!")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    # Save the uploaded file to disk temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Extract MFCC features
    mfcc = extract_mfcc("temp.wav")
    if mfcc is None:
        st.error("Error extracting features from the audio.")
    else:
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(device)  # shape: (1, time, features)

        with torch.no_grad():
            output = model(x)
            predicted = torch.argmax(output, dim=1).item()

        st.success(f"🎧 Predicted Emotion: **{emotion_labels[predicted]}**")

        # Optionally show audio
        st.audio("temp.wav")

    # Remove temporary file
    os.remove("temp.wav")
