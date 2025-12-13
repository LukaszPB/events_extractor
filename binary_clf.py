from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd

from data_processing import load_data, bag_of_words, SBERT_mini, SBERT

(X_train_raw, y_train), (X_test_raw, y_test), _ = load_data(binary=True, file_name='data/bigger_categorise_data.json')

X_train_bw, X_test_bw = bag_of_words(X_train_raw, X_test_raw)
X_train_SBERT_mini, X_test_SBERT_mini = SBERT_mini(X_train_raw, X_test_raw, 'data/big_data_embeddings_cache_mini.npz')
X_train_SBERT, X_test_SBERT = SBERT(X_train_raw, X_test_raw, 'data/big_data_embeddings_cache.npz')

# smote_bw = SMOTE(random_state=42)
# X_train_bw, y_train_bw = smote_bw.fit_resample(X_train_bw, y_train)

# smote_SBERT_mini = SMOTE(random_state=42)
# X_train_SBERT_mini, y_train_SBERT_mini = smote_SBERT_mini.fit_resample(X_train_SBERT_mini, y_train)

# smote_SBERT = SMOTE(random_state=42)
# X_train_SBERT, y_train_SBERT = smote_SBERT.fit_resample(X_train_SBERT, y_train)

# datasets = {
#     "Bag_of_Words": (X_train_bw, X_test_bw, y_train_bw),
#     "SBERT_mini": (X_train_SBERT_mini, X_test_SBERT_mini, y_train_SBERT_mini),
#     "SBERT": (X_train_SBERT, X_test_SBERT, y_train_SBERT)
# }

datasets = {
    "Bag_of_Words": (X_train_bw, X_test_bw),
    "SBERT_mini": (X_train_SBERT_mini, X_test_SBERT_mini),
    "SBERT": (X_train_SBERT, X_test_SBERT)
}

models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "SVM (LinearSVC)": LinearSVC(dual='auto', random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced'),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
}

results = []

for data_name, (X_train, X_test) in datasets.items():
    for model_name, model in models.items():
        if data_name == "Bag_of_Words" and model_name == "Naive Bayes":
            model = MultinomialNB()

        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)

        accuracy = model.score(X_test, y_test)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["NO_EVENT", "EVENT"], columns=["NO_EVENT", "EVENT"])
    
        results.append({
            "Method": data_name,
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro F1": f1,
            "Report": classification_report(y_test, y_pred, target_names=["NO_EVENT", "EVENT"], zero_division=0),
            "Confusion_Matrix": cm_df
        })


txt_filename = "experiments/baseline.txt"

df_results = pd.DataFrame(results)

df_sorted = df_results.sort_values(by=["Method", "Accuracy"], ascending=[True, False])
top5 = df_results.sort_values(by="Accuracy", ascending=False).head(5)

output = []

methods_order = ["Bag_of_Words", "SBERT_mini", "SBERT"]

for method in methods_order:
    output.append(f"\n================================ TESTOWANIE: {method} =============================")
    output.append("MODEL                     | ACCURACY   | PRECISION   | RECALL F1   | MACRO F1")
    output.append("--------------------------------------------------")
    
    sub = df_sorted[df_sorted["Method"] == method]
    for _, row in sub.iterrows():
        output.append(
            f"{row['Model']:<25} | {row['Accuracy']:.4f}     | {row['Precision']:.4f}      | {row['Recall']:.4f}      | {row['Macro F1']:.4f}"
        )

output.append("\n============================================================")
output.append("SZCZEGÓŁOWE RAPORTY KLASYFIKACJI")
output.append("============================================================\n")

for _, row in top5.iterrows():
    output.append(f"--- Metoda: {row['Method']} | Model: {row['Model']} ---")
    output.append(row["Report"])

    output.append("\nMACIERZ POMYŁEK (Wiersze=Prawda, Kolumny=Przewidywanie):")
    # Zamieniamy DataFrame na string tabelaryczny
    output.append(row["Confusion_Matrix"].to_string()) 
    output.append("\n" + "-"*60 + "\n")

with open("experiments/big_data_binary_clf.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Wynik zapisano do experiments/baseline.txt")