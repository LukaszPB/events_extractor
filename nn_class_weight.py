import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight # <--- NOWY IMPORT
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from data_processing import load_data, SBERT
from utils.nn_report import report

# --- KONFIGURACJA ---
NO_EVENT_LABEL = "NO_EVENT"
OUTPUT_FILENAME = "neural_network_class_weights_results.txt"

BEST_PARAMS = {
    "hidden_layers": [256, 128],
    "activation": "relu",
    "dropout": 0.4,
    "optimizer": "adam"
}

# Zamiast procentów usuwania, definiujemy tryby ważenia
# False = brak wag (standard), True = ważenie klas
WEIGHTING_OPTIONS = [False, True]

# --- 1. ŁADOWANIE DANYCH ---
(X_train_full, y_train_full), (_, _), encoder = load_data('data/biggest_categorise_data.json')
X_train_full, _ = SBERT(X_train_full, X_train_full, 'data/big_data_embeddings_cache2.npz')

target_label_name = NO_EVENT_LABEL
# Pobranie wszystkich unikalnych klas (potrzebne do wyliczenia wag)
classes_list = np.unique(y_train_full)

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=input_shape))
    for units in BEST_PARAMS["hidden_layers"]:
        model.add(layers.Dense(units, activation=BEST_PARAMS["activation"]))
        if BEST_PARAMS["dropout"] > 0:
            model.add(layers.Dropout(BEST_PARAMS["dropout"]))
    model.add(layers.Dense(len(encoder.classes_), activation="softmax"))
    model.compile(optimizer=BEST_PARAMS["optimizer"], loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

results = []

# --- 2. GŁÓWNA PĘTLA PO OPCJACH WAŻENIA ---
for use_weights in WEIGHTING_OPTIONS:
    mode_name = "With_Class_Weights" if use_weights else "Standard_No_Weights"
    print(f"\n--- Przetwarzanie w trybie: {mode_name} ---")
    
    # Nie modyfikujemy danych (nie usuwamy nic), używamy zawsze pełnego zbioru
    current_X = X_train_full
    current_y = y_train_full

    # B. Cross Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_true = []
    all_preds = [] 

    # Iteracja po foldach
    for i, (train_idx, val_idx) in enumerate(kfold.split(current_X, current_y)):
        X_tr, X_val = current_X[train_idx], current_X[val_idx]
        y_tr, y_val = current_y[train_idx], current_y[val_idx]
        
        # --- KLUCZOWA ZMIANA: OBLICZANIE WAG ---
        weights_dict = None
        if use_weights:
            # Obliczamy wagi tylko na podstawie zbioru treningowego w danym foldzie
            # 'balanced' wylicza to wg wzoru: n_samples / (n_classes * n_samples_j)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes_list,
                y=y_tr
            )
            # Keras wymaga słownika {numer_klasy: waga}
            weights_dict = dict(zip(classes_list, class_weights))
            
            if i == 0: # Wypisz przykładowe wagi tylko dla pierwszego foldu
                print(f"Wyliczone wagi klas: {weights_dict}")

        model = build_model((current_X.shape[1],))
        es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        
        # --- PRZEKAZANIE WAG DO TRENINGU ---
        model.fit(
            X_tr, y_tr, 
            validation_data=(X_val, y_val), 
            epochs=100, 
            batch_size=32, 
            verbose=0, 
            callbacks=[es],
            class_weight=weights_dict  # <--- Tutaj przekazujemy wagi
        )
        
        # Predykcja
        probs = model.predict(X_val, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        all_true.extend(y_val)
        all_preds.extend(preds)

    # C. Obliczanie metryk
    acc = accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_true, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    
    cm = confusion_matrix(all_true, all_preds)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    
    print(f"Wynik F1 Macro: {f1:.4f}")

    results.append({
        "Method": "SBERT+NN",
        "Model": mode_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Macro F1": f1,
        "Report": classification_report(all_true, all_preds, target_names=encoder.classes_, zero_division=0),
        "Confusion_Matrix": cm_df
    })

# --- 3. RAPORTOWANIE ---
report(results, OUTPUT_FILENAME)
print(f"Zapisano wyniki do {OUTPUT_FILENAME}")