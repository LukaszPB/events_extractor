import json
import time
import google.generativeai as genai
from google.api_core import retry

# --- KONFIGURACJA ---
API_KEY = "-"
INPUT_FILE = "data/politics_data.json"
OUTPUT_FILE = "data/politics_categorise_data.json"
BATCH_SIZE = 50

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
    model_name="gemma-3-27b-it", 
    generation_config={
        "temperature": 0.0
    }
)

def get_prompt(batch_data):
    return f"""
    You are a strictly formatted data classifier.
    
    ### INSTRUCTIONS
    You must output ONLY a valid JSON list. 
    DO NOT output code blocks (```json). 
    DO NOT output introductions or explanations.
    
    ### CATEGORY DEFINITIONS
    1. POLITICS: Government actions, legislation, elections, diplomacy, resignations.
    2. BUSINESS: Corporate actions (M&A), markets, regulatory fines, financial results.
    3. DISASTER: Physical destruction, fires, explosions, transport accidents, war acts.
    4. NO_EVENT: 
       - Future tense / Speculations ("Government plans to...", "Experts predict...").
       - Opinions / Commentary.
       - Historical background.
       - Petty crime (theft) unless involving VIPs.

    ### CRITICAL LOGIC RULES (Follow strictly)
    - FUTURE TENSE RULE: Promises, plans, and announcements are ALWAYS "NO_EVENT". Only factual, completed actions count.
    - PRIORITY RULE: If a sentence describes physical destruction or death, label as "DISASTER" (even if it involves a politician or company).
    - CONTEXT: Analyze ONLY the provided sentence.

    ### Examples (Few-Shot Learning):
    - Input: "The President signed the budget bill." -> Category: POLITICS, Trigger: "signed"
    - Input: "Analysts say the tax reform will boost the economy." -> Category: NO_EVENT, Trigger: "say/will"
    - Input: "The regulator fined the tech giant $5 million." -> Category: BUSINESS, Trigger: "fined"
    - Input: "Protesters set fire to the embassy." -> Category: DISASTER, Trigger: "set fire"
    - Input: "The thief stole a car from the parking lot." -> Category: NO_EVENT, Trigger: "stole"

    ### INPUT DATA
    {json.dumps(batch_data)}

    ### OUTPUT FORMAT
    Return a JSON list of objects matching this schema exactly:
    [
      {{
        "id": <integer_from_input>,
        "sentence": <string_from_input>,
        "category": "POLITICS" | "BUSINESS" | "DISASTER" | "NO_EVENT",
        "trigger_word": "<specific_word_justifying_decision>"
      }}
    ]
    """

def load_data():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Błąd ładowania danych: {e}")
        return []

def clean_gemma_response(text):
    """
    Funkcja czyszcząca. Gemma często dodaje ```json na początku.
    To usuwa te znaczniki, żeby json.loads() nie wyrzucił błędu.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

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
    print(f"Używany model: gemma-3-27b-it (Limit: 14.4k RPD / 30 RPM).")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("[\n")

    processed_count = 0

    for i in range(0, total_records, BATCH_SIZE):
        batch = clean_data[i : i + BATCH_SIZE]
        
        while True:
            try:
                prompt = get_prompt(batch)
                response = model.generate_content(prompt)

                # 1. Czyszczenie odpowiedzi (TO JEST KLUCZOWE DLA GEMMY)
                response_text = clean_gemma_response(response.text)
                
                try:
                    classified_batch = json.loads(response_text)
                except json.JSONDecodeError:
                    print("!!! Gemma zwróciła uszkodzony JSON. Ponawiam próbę...")
                    continue # Próbuj jeszcze raz tę samą paczkę

                # 2. Zapis
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    for index, record in enumerate(classified_batch):
                        if not (processed_count == 0 and index == 0):
                            f.write(",\n")
                        # Formatowanie Pretty JSON (każdy atrybut w nowej linii)
                        json_str = json.dumps(record, ensure_ascii=False, indent=4)
                        f.write(json_str)

                processed_count += len(batch)
                print(f"--> Przetworzono {processed_count} / {total_records}")
                break 
                 
            except Exception as e:
                error_msg = str(e)
                print(f"!!! Błąd przy partii {i}: {error_msg}")
                
                if "429" in error_msg:
                    print("Przekroczono limit. Czekam 60 sekund...")
                    time.sleep(60)
                else:
                    print("Inny błąd. Czekam 5 sekund...")
                    time.sleep(5)

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write("\n]")
    
    print(f"\nSukces.")

if __name__ == "__main__":
    process_and_save()