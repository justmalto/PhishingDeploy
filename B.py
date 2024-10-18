import streamlit as st
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained model and tokenizer
model = joblib.load('phishingmodel.pkl')
tokenizer = joblib.load('tokenizer.pkl')

MAX_SEQUENCE_LENGTH = 250

# Function to predict phishing text
def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    return np.argmax(pred, axis=1)[0]

# Streamlit app interface
st.title('Phishing Detector')

text = st.text_area('Enter the Text here')

if st.button('Detect'):
    label = predict(text)
    if label == 1:
        st.success("✅ This appears to be safe.")
    else:
        st.error("⚠️ This looks suspicious and might be a phishing attempt. ")
