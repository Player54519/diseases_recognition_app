import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# === 1. Charger le modèle fine-tuné ===
MODEL_PATH = "/home/vmeneghel/test_various/biobert-ner-finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

id2label = model.config.id2label  # {0: 'O', 1: 'B-Disease', ...}

# === 2. Prédiction avec alignement des sous-tokens ===
def predict_entities(text):
    encoding = tokenizer(text.split(), is_split_into_words=True, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    word_ids = encoding.word_ids()
    outputs = model(**encoding).logits
    predictions = torch.argmax(outputs, dim=2)[0].tolist()

    results = []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous_word_id:
            continue
        label = id2label[predictions[idx]]
        token = encoding.tokens()[idx]
        word = text.split()[word_id]
        results.append((word, label))
        previous_word_id = word_id
    return results

# === 3. Streamlit UI ===
st.title("🧬 Détection de maladies avec BioBERT fine-tuné")
st.write("Saisis une phrase biomédicale pour identifier les entités de type **Disease**.")

user_input = st.text_area("Texte d'entrée :", "The patient was diagnosed with diabetes and hypertension.", height=150)

if st.button("Analyser"):
    entities = predict_entities(user_input)

    st.markdown("### Résultat annoté :")
    html_output = ""
    for word, label in entities:
        if label.startswith("B-"):
            color = "#ffcccc"
            html_output += f"<span style='background-color:{color}; padding:3px; border-radius:4px; margin-right:3px'><strong>{word}</strong> <sub>{label}</sub></span> "
        elif label.startswith("I-"):
            color = "#ffe6e6"
            html_output += f"<span style='background-color:{color}; padding:3px; border-radius:4px; margin-right:3px'>{word}</span> "
        else:
            html_output += f"{word} "
    st.markdown(html_output, unsafe_allow_html=True)
