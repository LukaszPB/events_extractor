import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import StratifiedKFold
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
OUTPUT_FILENAME = "neural_network_balancing_impact.txt"

BEST_PARAMS = {
    "hidden_layers": [256, 128],
    "activation": "relu",
    "dropout": 0.4,
    "optimizer": "adam"
}

# Lista procentów do usunięcia z klasy NO_EVENT
# 0.0 = nie usuwamy nic (oryginalny zbiór)
# 0.5 = usuwamy 50% klasy NO_EVENT
# 0.8 = usuwamy 80% klasy NO_EVENT
DROP_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

# --- 1. ŁADOWANIE DANYCH (Tylko raz na początku) ---
# Ładujemy pełny zbiór, a potem będziemy go ciąć w pętli
(X_train_raw_full, y_train_full), (_, _), encoder = load_data('data/biggest_categorise_data.json')
X_train_full, _ = SBERT(X_train_raw_full, X_train_raw_full, 'data/big_data_embeddings_cache2.npz')

target_label_name = NO_EVENT_LABEL
target_label_id = encoder.transform([target_label_name])[0]
no_event_idx = list(encoder.classes_).index(NO_EVENT_LABEL)

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

# --- 2. GŁÓWNA PĘTLA PO PROPORCJACH USUWANIA ---
for drop_ratio in DROP_RATIOS:
    print(f"\n--- Przetwarzanie dla drop_ratio: {drop_ratio} (usuwanie {int(drop_ratio*100)}% {NO_EVENT_LABEL}) ---")
    
    # A. Logika usuwania danych
    current_X = X_train_full
    current_y = y_train_full
    
    if drop_ratio > 0.0:
        # Znajdź indeksy klasy NO_EVENT
        target_indices = np.where(y_train_full == target_label_id)[0]
        
        # Wylosuj indeksy do USUNIĘCIA
        np.random.seed(42) # Stałe ziarno dla powtarzalności w ramach tego samego ratio
        count_to_drop = int(len(target_indices) * drop_ratio)
        
        drop_indices = np.random.choice(
            target_indices, 
            size=count_to_drop, 
            replace=False
        )
        
        # Stwórz maskę
        mask = np.ones(len(y_train_full), dtype=bool)
        mask[drop_indices] = False
        
        # Zastosuj maskę
        current_X = X_train_full[mask]
        current_y = y_train_full[mask]
        
        print(f"Liczba próbek przed: {len(X_train_full)}, po: {len(current_X)}")
        print(f"Usunięto {count_to_drop} przykładów klasy {target_label_name}.")
    else:
        print("Brak redukcji danych (użycie pełnego zbioru).")

    # B. Cross Validation na zredukowanym zbiorze
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_true = []
    all_preds = [] # Tutaj zbieramy gotowe klasy, nie prawdopodobieństwa

    # Iteracja po foldach
    for i, (train_idx, val_idx) in enumerate(kfold.split(current_X, current_y)):
        X_tr, X_val = current_X[train_idx], current_X[val_idx]
        y_tr, y_val = current_y[train_idx], current_y[val_idx]
        
        model = build_model((current_X.shape[1],))
        es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0, callbacks=[es])
        
        # Predykcja
        probs = model.predict(X_val, verbose=0)
        preds = np.argmax(probs, axis=1) # Bierzemy klasę z najwyższym prawdopodobieństwem
        
        all_true.extend(y_val)
        all_preds.extend(preds)

    # C. Obliczanie metryk dla danego drop_ratio
    acc = accuracy_score(all_true, all_preds)
    prec = precision_score(all_true, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_true, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    
    cm = confusion_matrix(all_true, all_preds)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    
    # Nazwa modelu odzwierciedlająca stopień usunięcia danych
    model_name = f"NN_Drop_{int(drop_ratio*100)}%_{NO_EVENT_LABEL}"
    
    print(f"Wynik F1 Macro: {f1:.4f}")

    results.append({
        "Method": "SBERT+NN_Balancing",
        "Model": model_name,
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