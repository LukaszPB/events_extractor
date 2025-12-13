import pandas as pd

def report(results, filname):
    df_results = pd.DataFrame(results)

    df_sorted = df_results.sort_values(by=["Accuracy"], ascending=[False])

    output = []

    output.append("MODEL                     | ACCURACY   | PRECISION   | RECALL F1   | MACRO F1")
    output.append("--------------------------------------------------")
        
    for _, row in df_sorted.iterrows():
        output.append(
            f"{row['Model']:<25} | {row['Accuracy']:.4f}     | {row['Precision']:.4f}      | {row['Recall']:.4f}      | {row['Macro F1']:.4f}"
        )

    output.append("\n============================================================")
    output.append("SZCZEGÓŁOWE RAPORTY KLASYFIKACJI")
    output.append("============================================================\n")

    for _, row in df_sorted.iterrows():
        output.append(f"Model: {row['Model']} ---")
        output.append(row["Report"])

        output.append("\nMACIERZ POMYŁEK (Wiersze=Prawda, Kolumny=Przewidywanie):")
        # Zamieniamy DataFrame na string tabelaryczny
        output.append(row["Confusion_Matrix"].to_string()) 
        output.append("\n" + "-"*60 + "\n")

    with open(f"experiments/{filname}", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    print(f"Wynik zapisano do experiments/{filname}.txt")