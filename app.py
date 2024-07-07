import streamlit as st
import numpy as np
import pickle
from extract_feature import extract_feature
import os

model_path = 'emotion-audio-model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.title('Audio Emotion Detection')

uploaded_file = st.file_uploader("Upload an audio file to predict the emotion", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    
    feature=extract_feature(uploaded_file, mfcc=True, chroma=True, mel=True)
    features = np.array(feature).reshape(1, -1)
    prediction = model.predict(features)
    
    st.header(f"Predicted Emotion: {prediction[0]}")
