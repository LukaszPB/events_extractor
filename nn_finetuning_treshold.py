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
OUTPUT_DIR = './my_news_model'
DATA_PATH = 'data/biggest_categorise_data.json'
OUTPUT_FILENAME = "neural_network_finetuning_threshold.txt"
MAX_LEN = 64
TARGET_CLASS = "NO_EVENT"

def tokenize_data(sentences, tokenizer, max_len):
    """Pomocnicza funkcja do tokenizacji."""
    return tokenizer(
        sentences.tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors='tf'
    )

def get_predictions_for_threshold(probs, no_event_idx, threshold):
    """
    Logika decyzyjna:
    1. Jeśli p(NO_EVENT) >= threshold -> NO_EVENT
    2. W przeciwnym razie -> Klasa z najwyższym prawdopodobieństwem (pomijając NO_EVENT)
    """
    preds = []
    for p in probs:
        if p[no_event_idx] >= threshold:
            preds.append(no_event_idx)
        else:
            # Kopiujemy wektor prawdopodobieństw
            p_copy = p.copy()
            # Zerujemy (lub ustawiamy na -1) szansę NO_EVENT, żeby nie została wybrana w argmax
            p_copy[no_event_idx] = -1.0 
            # Wybieramy najlepszą z POZOSTAŁYCH kategorii
            preds.append(np.argmax(p_copy))
    return np.array(preds)

# --- START SKRYPTU ---

print(">>> [1/5] Wczytywanie danych...")
# WAŻNE: Wczytujemy dane, ale ignorujemy część testową (podłoga _).
# Interesuje nas tylko zachowanie modelu na danych, które widział (lub zbiorze walidacyjnym, jeśli tak load_data działa),
# aby dobrać parametry.
_, (X_train_raw, y_train), encoder = load_data(DATA_PATH, binary=False)

# Sprawdzenie indeksu klasy NO_EVENT
if TARGET_CLASS not in encoder.classes_:
    raise ValueError(f"Klasa '{TARGET_CLASS}' nie istnieje w encoderze!")
no_event_idx = np.where(encoder.classes_ == TARGET_CLASS)[0][0]
print(f"Index klasy '{TARGET_CLASS}': {no_event_idx}")

print(f">>> [2/5] Wczytywanie modelu z {OUTPUT_DIR}...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained(OUTPUT_DIR)
    model = TFDistilBertForSequenceClassification.from_pretrained(OUTPUT_DIR)
except OSError as e:
    print(f"Błąd krytyczny: Nie można załadować modelu z {OUTPUT_DIR}.\n{e}")
    exit(1)

print(">>> [3/5] Tokenizacja zbioru treningowego i generowanie surowych prawdopodobieństw...")
# 1. Tokenizacja
train_encodings = tokenize_data(X_train_raw, tokenizer, MAX_LEN)

# 2. Predykcja (robimy to RAZ dla całego zbioru, to najcięższa operacja)
# Zwraca logits -> zamieniamy na Softmax
pred_output = model.predict(train_encodings)
train_probs = tf.nn.softmax(pred_output.logits, axis=1).numpy()

print(f">>> [4/5] Eksperyment: Testowanie różnych progów dla '{TARGET_CLASS}'...")

thresholds_to_test = np.arange(0.1, 1.0, 0.05)
results = []

for t in thresholds_to_test:
    current_threshold = round(t, 2)
    
    # 1. Aplikacja logiki progowej na wynikach
    current_preds = get_predictions_for_threshold(train_probs, no_event_idx, current_threshold)
    
    # 2. Obliczanie metryk
    # Uwaga: Porównujemy predykcje z y_train (zbiór na którym kalibrujemy)
    acc = accuracy_score(y_train, current_preds)
    f1 = f1_score(y_train, current_preds, average="macro", zero_division=0)
    prec = precision_score(y_train, current_preds, average="macro", zero_division=0)
    rec = recall_score(y_train, current_preds, average="macro", zero_division=0)
    
    # Macierz pomyłek
    cm = confusion_matrix(y_train, current_preds)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    
    # Raport tekstowy (classification report)
    class_report = classification_report(y_train, current_preds, target_names=encoder.classes_, zero_division=0)
    
    # 3. Zapis wyniku do listy
    results.append({
        "Method": "Threshold_Experiment",
        "Model": f"DistilBERT (T={current_threshold:.2f})", # W nazwie modelu kodujemy próg
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Macro F1": f1,
        "Report": class_report,
        "Confusion_Matrix": cm_df
    })
    
    print(f"   -> Próg: {current_threshold:.2f} | F1 Macro: {f1:.4f}")

print(">>> [5/5] Generowanie raportu końcowego...")
report(results, OUTPUT_FILENAME)
print(f"Eksperyment zakończony. Wyniki dla wszystkich progów zapisano w: {OUTPUT_FILENAME}")