import json
import time
import google.generativeai as genai
from google.api_core import retry

API_KEY = "-"
INPUT_FILE = "data.json"
OUTPUT_FILE = "categorise_data.json"
BATCH_SIZE = 50

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.0,
        "response_mime_type": "application/json"
    }
)

def get_prompt(batch_data):
    """
    Tworzy prompt w języku angielskim z konkretnymi przykładami (Few-Shot).
    """
    return f"""
    You are an expert news analyst. Your task is to classify sentences into specific event categories based on their semantic meaning.
    
    Categories & Definitions:
    1. POLITICS:
       - Government actions, legislation, elections, diplomacy, resignations/appointments.
       - Entities: President, Parliament, Ministers, Parties.
    2. BUSINESS:
       - Corporate actions: M&A, bankruptcies, financial results.
       - Markets: Stock exchange, currency fluctuations.
       - Regulatory impact: Fines imposed on companies, antitrust rulings (focus on impact on business).
    3. DISASTER:
       - Physical destruction: Fires, explosions, natural disasters.
       - Major transport accidents (crashes, derailments).
       - War acts (bombings) if distinct from political decisions.
    4. NO_EVENT (Negative Class):
       - Future tense / Speculations ("Government plans to...", "Experts predict...").
       - Opinions / Commentary ("It is a good decision").
       - Historical background ("Founded in 1990...").
       - Petty crime (theft, burglary) unless involving VIPs.
       - Static descriptions.

    Critical Rules:
    - IGNORE the broader context of the article. Analyze ONLY the provided sentence.
    - FUTURE TENSE RULE: Promises, plans, and announcements are NO_EVENT. Only factual, completed, or ongoing actions count.
    - PRIORITY: If a sentence describes physical destruction/death, label as DISASTER (even if involving a CEO).

    Examples (Few-Shot Learning):
    - Input: "The President signed the budget bill." -> Category: POLITICS, Trigger: "signed"
    - Input: "Analysts say the tax reform will boost the economy." -> Category: NO_EVENT, Trigger: "say/will"
    - Input: "The regulator fined the tech giant $5 million." -> Category: BUSINESS, Trigger: "fined"
    - Input: "Protesters set fire to the embassy." -> Category: DISASTER, Trigger: "set fire"
    - Input: "The thief stole a car from the parking lot." -> Category: NO_EVENT, Trigger: "stole"

    DATA TO PROCESS:
    {json.dumps(batch_data)}

    OUTPUT FORMAT (JSON List):
    Return a valid JSON list of objects. Each object must have:
    - "id": (integer, same as input)
    - "sentence": (string, same as input)
    - "category": (string: "POLITICS", "BUSINESS", "DISASTER", or "NO_EVENT")
    - "trigger_word": (string, the specific verb or noun phrase that justified the decision)
    """

def load_data():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print("Błąd: Plik wejściowy musi zawierać listę JSON [ ... ]")
                return []
            return data
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {INPUT_FILE}")
        return []
    except json.JSONDecodeError:
        print(f"Błąd: Plik {INPUT_FILE} nie jest poprawnym JSON-em.")
        return []

def process_and_save():
    raw_data = load_data()
    if not raw_data:
        return

    clean_data = []
    for idx, record in enumerate(raw_data):
        clean_data.append({
            "id": record.get("id", idx + 1),
            "sentence": record.get("sentence", "")
        })
    
    total_records = len(clean_data)
    print(f"Załadowano {total_records} rekordów. Wielkość paczki: {BATCH_SIZE}.")
    print(f"Oszacowany czas dla {total_records} rekordów: ~{total_records/BATCH_SIZE * 5 / 60:.1f} minut.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("[\n")

    processed_count = 0

    for i in range(0, total_records, BATCH_SIZE):
        batch = clean_data[i : i + BATCH_SIZE]
        
        try:
            prompt = get_prompt(batch)
            response = model.generate_content(prompt)

            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            classified_batch = json.loads(response_text)
            
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for index, record in enumerate(classified_batch):
                    is_absolute_first = (processed_count == 0) and (index == 0)
                    
                    if not is_absolute_first:
                        f.write(",\n")

                    f.write("    " + json.dumps(record, ensure_ascii=False))

            processed_count += len(batch)
            print(f"--> Przetworzono {processed_count} / {total_records}")

            time.sleep(5)

        except Exception as e:
            print(f"!!! Błąd przy partii od indeksu {i}: {e}")
            print("Czekam 60 sekund przed ponowieniem (możliwy limit API)...")
            time.sleep(60) 

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("\n]")
    
    print(f"\nSukces")

if __name__ == "__main__":
    process_and_save()