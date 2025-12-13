import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
import itertools
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

# ---------------------------
# 1. Load data & embeddings
# ---------------------------
(X_train_raw, y_train), (_, _), encoder = load_data('data/bigger_categorise_data.json')
X_train, _ = SBERT(X_train_raw, X_train_raw, 'data/big_data_embeddings_cache.npz')

input_dim = X_train.shape[1]

# ---------------------------
# 2. Model builder
# ---------------------------
def build_model(hidden_layers, activation="relu", dropout=0.2, optimizer="adam"):
    # Use models.Sequential instead of keras.Sequential
    model = models.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    
    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    
    model.add(layers.Dense(len(encoder.classes_), activation="softmax"))
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
# ---------------------------
# 3. Parameter grid
# ---------------------------
hidden_structures = [
    [128, 64],
    [256, 128],
    [256, 256, 64],
    [512, 256, 128]
]


activations = ["relu", "tanh", "gelu"]
optimizers = ["adam", "rmsprop"]
dropouts = [0.0, 0.2, 0.4]

# tworzymy wszystkie kombinacje
keys = ["hidden_layers", "activation", "optimizer", "dropout"]
values = [hidden_structures, activations, optimizers, dropouts]

param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

data_name = "SBERT"  # metoda, jak w Twoim formacie

# ---------------------------
# 4. Cross-validation
# ---------------------------
for idx, params in enumerate(param_grid, start=1):
    model_name = f"{params['activation']}_{params['optimizer']}_{params['dropout']}_" + "x".join(map(str, params['hidden_layers']))
    print(f"\n=== Model {idx}/{len(param_grid)}: {model_name}")
    
    all_true = []
    all_pred = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), start=1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model = build_model(**params)
        
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
        
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=32,
            verbose=0,
            callbacks=[es]
        )
        
        y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        
        all_true.extend(y_val)
        all_pred.extend(y_val_pred)
    
    # Metrics based on all CV folds combined
    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, average="macro", zero_division=0)
    recall = recall_score(all_true, all_pred, average="macro", zero_division=0)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    cm = confusion_matrix(all_true, all_pred)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    
    results.append({
        "Method": data_name,
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": f1,
        "Report": classification_report(all_true, all_pred, target_names=encoder.classes_, zero_division=0),
        "Confusion_Matrix": cm_df
    })

print("\nCross-validation complete. Results collected for all parameter sets.")

# ---------------------------
# 5. Optional: zapis wyników do pliku
# ---------------------------
report(results, "neural_network.txt")
