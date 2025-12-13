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

NO_EVENT_LABEL = "NO_EVENT"
OUTPUT_FILENAME = "neural_network_thresholds2.txt"

BEST_PARAMS = {
    "hidden_layers": [256, 128],
    "activation": "relu",
    "dropout": 0.4,
    "optimizer": "adam"
}

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

(X_train_raw, y_train), (_, _), encoder = load_data('data/biggest_categorise_data.json')
X_train, _ = SBERT(X_train_raw, X_train_raw, 'data/big_data_embeddings_cache2.npz')

target_label_name = "NO_EVENT"  # Zmień na nazwę swojej klasy większościowej
target_label_id = encoder.transform([target_label_name])[0]

# 2. Znajdź wszystkie indeksy (wiersze), gdzie występuje ta klasa
target_indices = np.where(y_train == target_label_id)[0]

# 3. Wylosuj połowę tych indeksów do USUNIĘCIA
np.random.seed(42)  # Dla powtarzalności wyników
drop_indices = np.random.choice(
    target_indices, 
    size=int(len(target_indices) * 0.8),  # 0.5 oznacza usunięcie 50%
    replace=False
)

# 4. Stwórz maskę boolowską (Tablica True/False)
# Na początku wszyscy mają True (zostają)
mask = np.ones(len(y_train), dtype=bool)
# Ustawiamy False (usuń) dla wylosowanych indeksów
mask[drop_indices] = False

# 5. Zastosuj maskę do wszystkich zbiorów danych
X_train_reduced = X_train[mask]
y_train_reduced = y_train[mask]

# Jeśli X_train_raw to lista, musisz ją najpierw zamienić na array, żeby użyć maski
X_train_raw_reduced = np.array(X_train_raw)[mask]

# --- Weryfikacja ---
print(f"Przed: {len(X_train)}")
print(f"Po: {len(X_train_reduced)}")
print(f"Usunięto {len(drop_indices)} przykładów klasy {target_label_name}.")

# Nadpisz zmienne, jeśli chcesz kontynuować z nowym zbiorem
X_train = X_train_reduced
y_train = y_train_reduced
X_train_raw = X_train_raw_reduced


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

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_true = []
all_probs = []

for train_idx, val_idx in kfold.split(X_train, y_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    model = build_model((X_train.shape[1],))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0, callbacks=[es])
    
    probs = model.predict(X_val, verbose=0)
    all_true.extend(y_val)
    all_probs.extend(probs)

all_true = np.array(all_true)
all_probs = np.array(all_probs)

results = []

for threshold in THRESHOLDS:
    final_preds = []
    
    for probs in all_probs:
        pred_idx = np.argmax(probs)
        
        if threshold > 0.0 and pred_idx == no_event_idx:
            if probs[no_event_idx] < threshold:
                temp_probs = probs.copy()
                temp_probs[no_event_idx] = -1.0
                pred_idx = np.argmax(temp_probs)
        
        final_preds.append(pred_idx)

    acc = accuracy_score(all_true, final_preds)
    prec = precision_score(all_true, final_preds, average="macro", zero_division=0)
    rec = recall_score(all_true, final_preds, average="macro", zero_division=0)
    f1 = f1_score(all_true, final_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_true, final_preds)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    
    model_name = f"NN_Best_Threshold_{threshold}" if threshold > 0 else "NN_Best_Default"
    
    results.append({
        "Method": "SBERT+NN",
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Macro F1": f1,
        "Report": classification_report(all_true, final_preds, target_names=encoder.classes_, zero_division=0),
        "Confusion_Matrix": cm_df
    })

report(results, OUTPUT_FILENAME)