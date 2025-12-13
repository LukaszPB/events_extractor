import requests
from bs4 import BeautifulSoup
import nltk
import re
import unicodedata
import pandas as pd

# Upewnij się, że tokenizery są pobrane
try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')
try: nltk.data.find('tokenizers/punkt_tab')
except LookupError: nltk.download('punkt_tab')

output_data = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# Konfiguracja - od której strony zaczynać w każdej kategorii
START_FROM_PAGE = 8 

sources = [
    (
        "https://en.wikinews.org/wiki/Category:Economy_and_business", 
        "BUSINESS", 
        ["crime", "law", "disaster", "accident", "death", "politics", "election", "entertainment"],
        0 # Ustaw cel (ARTICLES_NUM) zgodnie z potrzebami
    ),
    (
        "https://en.wikinews.org/wiki/Category:Politics_and_conflicts", 
        "POLITICS",
        ["sport", "entertainment", "disaster", "accident", "obituary", "business", "economy"],
        300
    ),
    (
        "https://en.wikinews.org/wiki/Category:Disasters_and_accidents", 
        "DISASTER",
        [],
        0
    )
]

print(f"Rozpoczynam pracę. Start od strony: {START_FROM_PAGE}...")

for start_url, label, blacklist_tags, ARTICLES_NUM in sources:
    current_url = start_url
    collected_in_category = 0
    page_number = 1
    
    print(f"\n--- KATEGORIA: {label} (Cel: {ARTICLES_NUM}) ---")
    
    while collected_in_category < ARTICLES_NUM:
        # Informacja o stanie
        if page_number < START_FROM_PAGE:
            print(f"Pominiecie strony {page_number} (Szukam strony {START_FROM_PAGE})...")
        else:
            print(f"Pobieranie listy artykułów ze strony {page_number}...")

        try:
            response = requests.get(current_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # --- BLOK 1: SZYBKIE PRZEWIJANIE DO START_FROM_PAGE ---
            if page_number < START_FROM_PAGE:
                # Szukamy tylko linku "next", żeby przewinąć
                next_link = soup.find('a', string=re.compile(r"next page|next 200", re.IGNORECASE))
                
                if next_link:
                    current_url = "https://en.wikinews.org" + next_link['href']
                    page_number += 1
                    continue # <--- Wraca na początek pętli while, pomija resztę kodu
                else:
                    print(" -> Nie można dotrzeć do wybranej strony startowej (brak kolejnych stron).")
                    break
            # ------------------------------------------------------

            # --- BLOK 2: WŁAŚCIWE POBIERANIE (DLA page_number >= 8) ---
            links = soup.select('div#mw-pages li a')
            
            if not links:
                print(" ! Nie znaleziono żadnych artykułów na tej stronie.")
                break

            for link in links:
                if collected_in_category >= ARTICLES_NUM: break
                if "Category:" in link['href']: continue
                
                # --- Pobieranie treści artykułu ---
                try:
                    full_url = "https://en.wikinews.org" + link['href']
                    art_resp = requests.get(full_url, headers=headers)
                    art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                    
                    # Sprawdzanie tagów (czarna lista)
                    cat_links = art_soup.select('div#catlinks li a')
                    article_tags = [t.text.lower() for t in cat_links]
                    
                    if any(bad in tag for tag in article_tags for bad in blacklist_tags):
                        continue

                    # Pobieranie treści
                    content_div = art_soup.find('div', {'class': 'mw-parser-output'})
                    if not content_div: continue

                    raw_text = " ".join([p.text.strip() for p in content_div.find_all('p')])
                    text = unicodedata.normalize('NFKD', raw_text)
                    text = re.sub(r'\[\d+\]', '', text)
                    text = text.replace('\u2014', '-').replace('"', "'").strip()

                    sentences_added = 0
                    for sent in nltk.sent_tokenize(text):
                        clean = sent.strip()
                        if len(clean) > 20 and clean[0].isupper():
                            output_data.append({"sentence": clean, "category": label})
                            sentences_added += 1
                    
                    if sentences_added > 0:
                        collected_in_category += 1
                        
                except Exception:
                    continue
                # --- Koniec artykułu ---

            print(f"   Status: Zebrano {collected_in_category}/{ARTICLES_NUM} artykułów.")

            # --- PAGINACJA (DLA NORMALNEGO TRYBU) ---
            next_link = soup.find('a', string=re.compile(r"next page|next 200", re.IGNORECASE))
            
            if next_link and collected_in_category < ARTICLES_NUM:
                current_url = "https://en.wikinews.org" + next_link['href']
                page_number += 1
                print(f" -> Przechodzę do strony {page_number}")
            else:
                print(" -> Nie znaleziono linku 'next page' lub osiągnięto cel. Koniec tej kategorii.")
                break
                
        except Exception as e:
            print(f"Błąd krytyczny pętli: {e}")
            break

df = pd.DataFrame(output_data)
# Usuwanie duplikatów
df.drop_duplicates(subset=['sentence'], keep='first', inplace=True)

# Zapis do pliku
df.to_json('data/politics_data.json', orient='records', force_ascii=False, indent=4)
print(f"\nZakończono! Łącznie zapisano {len(df)} unikalnych zdań.")