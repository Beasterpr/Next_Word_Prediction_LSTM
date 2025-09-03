import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras import backend as K
import numpy as np
import pickle

@st.cache_resource
def load_tokenizer():
    with open("token.pkl", "rb") as f:
        return pickle.load(f)
    
@st.cache_resource
def model_load():
    model = load_model(
        "model_export.h5",
        custom_objects={"LSTM": LSTM, "Bidirectional": Bidirectional},
        compile=False
    )
    return model

token = load_tokenizer()
model = model_load()



st.title("🔮 Next Word Prediction")
st.caption("🗨️ Trained on DailyDialog - best at predicting next words in normal conversations (e.g., greetings, opinions, questions, daily life topics).")




input_text = st.text_input("Type your sentence here...", placeholder="The quick brown fox")

def predict_top10(text):
    text = text.strip().lower()
    seq = token.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=30, padding='pre')
    prediction = model.predict(seq)[0]   
    top10_idx = prediction.argsort()[-10:][::-1]
    index_word = {i: w for w, i in token.word_index.items()}
    results = [(index_word.get(i, "<OOV>"), prediction[i]) for i in top10_idx]
    return results




if st.button("🚀 Get Suggestions", type="primary"):
    with st.spinner("Generating predictions..."):
        predictions = predict_top10(input_text)
        for i, (word, _) in enumerate(predictions, 1):
                st.write(f"{i}. {word}")
        
        if predictions:
            st.subheader("📝 Top Word Suggestions:")

with st.expander("ℹ️ About the Model"):
    st.markdown("""
    **Model Overview:**
    - **Type:** LSTM-based Neural Network
    - **Architecture:** Bidirectional LSTM layers
    - **Input:** Sequence of tokens (words) from user text
    - **Output:** Probabilities of next words in vocabulary
    - **Training Data:** DailyDialog dataset (general conversational text)
    - **Purpose:** Predicts the next word in a sentence for normal conversation
    - **Max Sequence Length:** 30 tokens
    - **Tokenizer:** Word-level, saved as `token.pkl`

    **How it Works:**
    1. User types a sentence.
    2. Text is converted into tokens using the tokenizer.
    3. Sequence is padded to the model's input length.
    4. Model predicts probabilities for all words in vocabulary.
    5. Top 10 words are shown to the user.
    """)

            
          
                
      


