import pandas as pd
import spacy
import json

INPUT_FILE = "data/who-what.json"
OUTPUT_DETAILS = "experiments_who_what/detailed_results.json"
OUTPUT_SUMMARY = "experiments_who_what/summary.json"

nlp = spacy.load("en_core_web_sm")

def extract_information(row):
    sentence_text = row['sentence']
    doc = nlp(sentence_text)
    
    extracted = {
        "who": "",
        "trigger": "", 
        "what": "", 
        "where": "", 
        "when": ""
    }

    root = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), 
                next((t for t in doc if t.pos_ == "VERB"), None))

    if root:
        extracted["trigger"] = root.text
        
        subj_deps = {"nsubj", "agent", "nsubjpass"}
        obj_deps = {"obj", "dobj", "pobj"}

        for child in root.children:
            phrase = child.subtree
            phrase_text = "".join([t.text_with_ws for t in phrase]).strip()
            
            if child.dep_ in subj_deps:
                extracted["who"] = phrase_text
            elif child.dep_ in obj_deps:
                extracted["what"] = phrase_text

    extracted["where"] = ", ".join([ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]])
    extracted["when"] = ", ".join([ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]])

    return pd.Series(extracted)

df = pd.read_json(INPUT_FILE)

predictions_df = df.apply(extract_information, axis=1)

full_report = []
fields = ["who", "trigger", "what", "where", "when"]
correct_counts = {field: 0 for field in fields}

for idx, row in df.iterrows():
    comparison_entry = {
        "sentence": row["sentence"],
        "category": row.get("category", "N/A"),
        "analysis": {}
    }
    
    for field in fields:
        pred_val = str(predictions_df.at[idx, field]).lower().strip()
        orig_val = str(row.get(field, "")).lower().strip() if pd.notna(row.get(field)) else ""
        
        is_correct = (pred_val == orig_val)
        if is_correct:
            correct_counts[field] += 1
            
        comparison_entry["analysis"][field] = {
            "predicted": predictions_df.at[idx, field],
            "actual": row.get(field, None),
            "is_correct": is_correct
        }
    
    full_report.append(comparison_entry)

total = len(df)
summary = {
    "total_sentences": total,
    "accuracy_per_field": {field: f"{(count/total)*100:.2f}%" for field, count in correct_counts.items()},
    "correct_counts": correct_counts
}

with open(OUTPUT_DETAILS, "w", encoding="utf-8") as f:
    json.dump(full_report, f, indent=4, ensure_ascii=False)

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print(f"Przetwarzanie zakończone. Statystyki:")
for field, acc in summary["accuracy_per_field"].items():
    print(f"- {field}: {acc}")