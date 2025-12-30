import os
import logging
import joblib
import nltk
import spacy
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from typing import Dict, Any, List, Optional

# --- KONFIGURACJA ---
class Config:
    MODEL_DIR = './model'
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    TARGET_CLASS = "NO_EVENT"
    THRESHOLD = 0.70
    MAX_LEN = 64
    PORT = 5000
    HOST = '0.0.0.0'

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- ZASOBY ---
class ModelAssets:
    def __init__(self):
        self.model: Optional[TFDistilBertForSequenceClassification] = None
        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.encoder: Any = None
        self.nlp: Any = None
        self.no_event_idx: int = -1

    def load(self):
        logging.info(">>> Loading ML model and resources...")
        
        # NLTK dependencies
        nltk.download('punkt', quiet=True)

        try:
            # 1. BERT & Tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_DIR)
            self.model = TFDistilBertForSequenceClassification.from_pretrained(Config.MODEL_DIR)
            
            # 2. Label Encoder
            self.encoder = joblib.load(Config.LABEL_ENCODER_PATH)
            classes = self.encoder.classes_
            
            if Config.TARGET_CLASS in classes:
                self.no_event_idx = np.where(classes == Config.TARGET_CLASS)[0][0]
            else:
                logging.warning(f"Class {Config.TARGET_CLASS} not found in encoder!")

            # 3. Spacy
            self.nlp = spacy.load("en_core_web_sm")
            logging.info(">>> Resources loaded successfully.")
            
        except Exception as e:
            logging.error(f"Critical error during resource loading: {e}")
            raise e

assets = ModelAssets()

# --- LOGIKA ANALIZY ---

def extract_information(sentence: str) -> Dict[str, str]:
    """Ekstrakcja informacji (NER + Dependency Parsing) przy użyciu SpaCy."""
    doc = assets.nlp(sentence)
    
    extracted = {
        "who": "",
        "trigger": "", 
        "what": "", 
        "where": "", 
        "when": ""
    }

    # Szukanie głównego czasownika (ROOT)
    root = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), 
                next((t for t in doc if t.pos_ == "VERB"), None))

    if root:
        extracted["trigger"] = root.text
        
        subj_deps = {"nsubj", "agent", "nsubjpass"}
        obj_deps = {"obj", "dobj", "pobj"}

        for child in root.children:
            phrase_text = "".join([t.text_with_ws for t in child.subtree]).strip()
            
            if child.dep_ in subj_deps:
                extracted["who"] = phrase_text
            elif child.dep_ in obj_deps:
                extracted["what"] = phrase_text

    # Ekstrakcja encji nazwanych (GPE=Geopolityczne, FAC=Budynki, DATE=Daty)
    extracted["where"] = ", ".join([ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]])
    extracted["when"] = ", ".join([ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]])

    return extracted

def predict_sentence_class(sentence: str):
    """Klasyfikacja zdania z uwzględnieniem progu ufności dla NO_EVENT."""
    inputs = assets.tokenizer(
        [sentence], 
        truncation=True, 
        padding=True, 
        max_length=Config.MAX_LEN, 
        return_tensors='tf'
    )
    
    logits = assets.model(inputs).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    
    # Logika progowa dla klasy neutralnej
    if assets.no_event_idx != -1:
        score_no_event = probs[assets.no_event_idx]
        if score_no_event >= Config.THRESHOLD:
            predicted_idx = assets.no_event_idx
        else:
            probs_copy = probs.copy()
            probs_copy[assets.no_event_idx] = -1.0
            predicted_idx = np.argmax(probs_copy)
    else:
        predicted_idx = np.argmax(probs)
        
    label = assets.encoder.classes_[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    return label, confidence

# --- ENDPOINTY ---

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    raw_text = data['text']
    sentences = nltk.sent_tokenize(raw_text)
    results = []
    
    for sent in sentences:
        label, confidence = predict_sentence_class(sent)
        
        # Podstawowy rekord
        record = {
            "sentence": sent,
            "event_type": label,
            "meta": {"confidence": round(confidence, 4)}
        }
        
        # Ekstrakcja szczegółów tylko dla istotnych zdarzeń
        if label != Config.TARGET_CLASS:
            info = extract_information(sent)
            record.update(info)
        else:
            # Wypełnienie pustymi wartościami dla zachowania spójności struktury JSON
            record.update({k: None for k in ["who", "trigger", "what", "where", "when"]})
            
        results.append(record)
        
    return jsonify({"results": results})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "model_loaded": assets.model is not None
    })

if __name__ == '__main__':
    assets.load()
    app.run(host=Config.HOST, port=Config.PORT, debug=False)