
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Load CRF model and GloVe vectors
@st.cache_resource
def load_model():
    crf_model = joblib.load("crf_hybrid_model.pkl")
    glove = joblib.load("glove_vectors.pkl")
    return crf_model, glove

crf_hybrid, glove = load_model()

# Feature extraction function
def glove_hybrid_features(sentence):
    features = []
    for i, token in enumerate(sentence):
        word = token['token']
        pos = ''
        vec = glove[word].numpy() if word in glove.stoi else np.zeros(glove.dim)

        vec_dict = {f"dim_{i}": float(v) for i, v in enumerate(vec)}
        vec_dict.update({
            'is_title': word.istitle(),
            'is_upper': word.isupper(),
            'is_digit': word.isdigit(),
            'pos_tag': pos
        })

        if i > 0:
            vec_dict['prev_token'] = sentence[i - 1]['token'].lower()
        if i < len(sentence) - 1:
            vec_dict['next_token'] = sentence[i + 1]['token'].lower()

        features.append(vec_dict)
    return features

# Logging function
def log_search(input_text, predictions):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_text,
        "output_tags": " ".join(predictions)
    }

    try:
        log_df = pd.read_csv("log.csv")
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
    except FileNotFoundError:
        log_df = pd.DataFrame([log_entry])

    log_df.to_csv("log.csv", index=False)

# Streamlit UI
st.title("Abbreviation & Long-form BIO Tagger (CRF Model)")
input_text = st.text_input("Enter a biomedical sentence:")

if input_text:
    tokens = input_text.split()
    sent_struct = [{'token': tok} for tok in tokens]
    feats = glove_hybrid_features(sent_struct)
    preds = crf_hybrid.predict_single(feats)

    st.markdown("### Predicted BIO Tags")
    for tok, tag in zip(tokens, preds):
        st.write(f"`{tok}` → {tag}")

    # Log the interaction
    log_search(input_text, preds)
