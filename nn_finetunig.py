import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# --- Importy Twoich modułów ---
from data_processing import load_data
from utils.nn_report import report

# --- Konfiguracja ---
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
OUTPUT_DIR = './my_news_model'
OUTPUT_FILENAME = "neural_network_finetuning.txt"
DATA_PATH = 'data/biggest_categorise_data.json'

# Globalna lista na wyniki
results = []

def tokenize_data(sentences, tokenizer, max_len):
    """
    Pomocnicza funkcja do tokenizacji tekstu dla modelu BERT.
    """
    return tokenizer(
        sentences.tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors='tf'
    )

def run_pipeline():
    """
    Główna funkcja procesująca: Wczytanie -> Trening -> Ewaluacja -> Raport -> Zapis.
    """
    # 1. Wczytanie danych
    print(">>> [1/6] Wczytywanie danych...")
    (X_train_raw, y_train), (X_test_raw, y_test), encoder = load_data(DATA_PATH, binary=False)

    # 2. Przygotowanie danych (Tokenizacja)
    print(">>> [2/6] Tokenizacja danych...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenize_data(X_train_raw, tokenizer, MAX_LEN)
    test_encodings = tokenize_data(X_test_raw, tokenizer, MAX_LEN)

    # Tworzenie obiektów tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # 3. Inicjalizacja i trening modelu
    print(f">>> [3/6] Inicjalizacja modelu {MODEL_NAME} i trening...")
    
    # UWAGA: use_safetensors=False naprawia błąd "builtins.safe_open object is not iterable"
    model = TFDistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(encoder.classes_),
        use_safetensors=False  
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS)

    # 4. Ewaluacja (Generowanie predykcji)
    print(">>> [4/6] Generowanie predykcji i obliczanie metryk...")
    y_pred_logits = model.predict(test_dataset).logits
    y_pred_idx = np.argmax(y_pred_logits, axis=1)

    # Obliczanie metryk
    accuracy = accuracy_score(y_test, y_pred_idx)
    precision = precision_score(y_test, y_pred_idx, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred_idx, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred_idx, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred_idx)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    clf_report = classification_report(y_test, y_pred_idx, target_names=encoder.classes_, zero_division=0)

    # Dodanie wyników do listy
    results.append({
        "Method": "News_Dataset",
        "Model": "DistilBERT_FineTuned",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": f1,
        "Report": clf_report,
        "Confusion_Matrix": cm_df
    })
    
    print(f"    -> Accuracy: {accuracy:.4f}")

    # 5. Raportowanie (Zapis do pliku)
    print(f">>> [5/6] Zapisywanie raportu do pliku: {OUTPUT_FILENAME}...")
    # Przekazujemy nazwę pliku jako drugi argument
    report(results, OUTPUT_FILENAME)

    # 6. Zapis modelu i artefaktów
    print(f">>> [6/6] Zapisywanie modelu do folderu: {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    joblib.dump(encoder, os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))
    
    print(">>> Proces zakończony sukcesem.")

# --- Klasa do Inferencji ---
class NewsPredictor:
    """
    Klasa służąca do wczytania zapisanego modelu i wykonywania predykcji na nowym tekście.
    """
    def __init__(self, model_path):
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.encoder = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))
        
    def predict_proba(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="tf", 
            truncation=True, 
            padding=True, 
            max_length=MAX_LEN
        )
        logits = self.model(inputs).logits
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        return {cls: float(probs[i]) for i, cls in enumerate(self.encoder.classes_)}

# --- Uruchomienie ---
if __name__ == "__main__":
    # 1. Uruchomienie głównego potoku
    run_pipeline()
    
    # 2. Wyświetlenie podsumowania w konsoli
    if results:
        results_df = pd.DataFrame(results)
        print("\n--- ZBIORCZE WYNIKI ---")
        print(results_df[['Model', 'Accuracy', 'Macro F1']])
    
    # 3. Szybki test inferencji
    print("\n--- TEST INFERENCJI (Przykładowe zdanie) ---")
    try:
        predictor = NewsPredictor(OUTPUT_DIR)
        sample_text = "The government announced new tax reforms today."
        probs = predictor.predict_proba(sample_text)
        print(f"Tekst: '{sample_text}'")
        print("Prawdopodobieństwa:", probs)
    except Exception as e:
        print(f"Błąd podczas testu inferencji: {e}")