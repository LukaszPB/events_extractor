import pandas as pd

def report(results, filname):
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

    with open(f"experiments/{filname}", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    print(f"Wynik zapisano do experiments/{filname}.txt")