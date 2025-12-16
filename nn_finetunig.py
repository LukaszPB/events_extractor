import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from tf_keras.optimizers import Adam
from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.callbacks import EarlyStopping

from data_processing import load_data
from utils.nn_report import report

MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 4
VAL_SIZE = 0.15

OUTPUT_DIR = './my_news_model'
OUTPUT_FILENAME = "neural_network_finetuning_full.txt"
DATA_PATH = 'data/biggest_categorise_data.json'

TARGET_CLASS = "NO_EVENT"
THRESHOLDS = np.arange(0.1, 1.0, 0.05)

results = []

def tokenize_data(sentences, tokenizer, max_len):
    return tokenizer(
        sentences.tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors='tf'
    )

def get_predictions_for_threshold(probs, no_event_idx, threshold):
    preds = []
    for p in probs:
        if p[no_event_idx] >= threshold:
            preds.append(no_event_idx)
        else:
            p_copy = p.copy()
            p_copy[no_event_idx] = -1.0 
            preds.append(np.argmax(p_copy))
    return np.array(preds)

def calculate_metrics_and_append(y_true, y_pred, encoder, model_name, method_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    clf_report = classification_report(y_true, y_pred, target_names=encoder.classes_, zero_division=0)

    results.append({
        "Method": method_name,
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Macro F1": f1,
        "Report": clf_report,
        "Confusion_Matrix": cm_df
    })
    return f1

def run_pipeline():

    print(">>> [1/7] Wczytywanie danych...")
    (X_train_full, y_train_full), (X_test, y_test), encoder = load_data(DATA_PATH, binary=False)

    print(f">>> [2/7] Wydzielanie zbioru walidacyjnego ({VAL_SIZE*100}%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=VAL_SIZE, 
        random_state=42, 
        stratify=y_train_full
    )
    
    print(f"    Liczebności -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print(">>> [3/7] Tokenizacja danych...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenize_data(X_train, tokenizer, MAX_LEN)
    val_encodings = tokenize_data(X_val, tokenizer, MAX_LEN)
    test_encodings = tokenize_data(X_test, tokenizer, MAX_LEN)

    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    print(f">>> [4/7] Trening modelu {MODEL_NAME} z Early Stopping...")
    model = TFDistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(encoder.classes_),
        use_safetensors=False
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=1,
        restore_best_weights=True,
        verbose=1
    )
    
    model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=EPOCHS, 
        callbacks=[early_stopping]
    )

    print(f">>> [5/7] Zapisywanie modelu (najlepsza wersja) do: {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    joblib.dump(encoder, os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))

    print(">>> [6/7] Obliczanie wyniku bazowego (argmax) na zbiorze Testowym...")
    y_pred_logits = model.predict(test_dataset).logits
    y_pred_idx = np.argmax(y_pred_logits, axis=1)
    
    calculate_metrics_and_append(
        y_test, y_pred_idx, encoder, 
        model_name="DistilBERT_Standard", 
        method_name="Baseline_TestSet"
    )

    print(f">>> [7/7] Optymalizacja progu dla '{TARGET_CLASS}' na zbiorze WALIDACYJNYM...")
    
    if TARGET_CLASS not in encoder.classes_:
        raise ValueError(f"Klasa '{TARGET_CLASS}' nie istnieje w encoderze!")
    no_event_idx = np.where(encoder.classes_ == TARGET_CLASS)[0][0]
    
    print("    Generowanie prawdopodobieństw dla zbioru walidacyjnego...")
    val_logits = model.predict(val_dataset).logits
    val_probs = tf.nn.softmax(val_logits, axis=1).numpy()
    
    print("    Iteracja po progach...")
    best_f1 = 0
    best_threshold = 0
    
    for t in THRESHOLDS:
        current_threshold = round(t, 2)
        val_preds_t = get_predictions_for_threshold(val_probs, no_event_idx, current_threshold)
        
        current_f1 = calculate_metrics_and_append(
            y_val, val_preds_t, encoder,
            model_name=f"DistilBERT (T={current_threshold:.2f})",
            method_name="Threshold_Validation"
        )
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = current_threshold
            
        print(f"    -> Próg: {current_threshold:.2f} | Val F1 Macro: {current_f1:.4f}")

    print(f"\n>>> Najlepszy próg na walidacji: {best_threshold} (F1: {best_f1:.4f})")

    print(f">>> Zapisywanie raportu do pliku: {OUTPUT_FILENAME}...")
    report(results, OUTPUT_FILENAME)
    print(">>> Proces zakończony.")

if __name__ == "__main__":
    run_pipeline()