import requests
from bs4 import BeautifulSoup
import nltk
import re
import unicodedata
import pandas as pd

try: nltk.data.find('tokenizers/punkt')
except LookupError: nltk.download('punkt')

output_data = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

ARTICLES_NUM = 150

sources = [
    (
        "https://en.wikinews.org/wiki/Category:Economy_and_business", 
        "BUSINESS", 
        # Wykluczone kategorie
        ["crime", "law", "disaster", "accident", "death", "politics", "election", "entertainment"]
    ),
    (
        "https://en.wikinews.org/wiki/Category:Politics_and_conflicts", 
        "POLITICS",
        ["sport", "entertainment", "disaster", "accident", "obituary", "business", "economy"]
    ),
    (
        "https://en.wikinews.org/wiki/Category:Disasters_and_accidents", 
        "DISASTER",
        []
    )
]

print("Pobieranie danych (BUSINESS, POLITICS, DISASTER)...")

for url, label, blacklist_tags in sources:
    try:
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
        links = soup.select('div#mw-pages li a')[:5*ARTICLES_NUM]

        collected = 0
        for link in links:
            if collected >= ARTICLES_NUM: break
            if "Category:" in link['href']: continue
            
            try:
                art_resp = requests.get("https://en.wikinews.org" + link['href'], headers=headers)
                art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                
                cat_links = art_soup.select('div#catlinks li a')
                article_tags = [tag.text.lower() for tag in cat_links]
                
                is_dirty = False
                for tag in article_tags:
                    if any(bad_word in tag for bad_word in blacklist_tags):
                        is_dirty = True
                        print(f"   POMINIĘTO: '{link.text[:30]}...' (Kolizja: {tag})")
                        break
                
                if is_dirty: continue
              
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
                        output_data.append({
                            "sentence": clean,
                            "category": label
                        })
                        sentences_added += 1
                
                if sentences_added > 0:
                    collected += 1
                    print(f"   [{label}] Dodano: {link.text[:40]}...")
                    
            except Exception: continue
    except Exception: continue

df = pd.DataFrame(output_data)
df.drop_duplicates(subset=['sentence'], keep='first', inplace=True)

df.to_json(
    'data.json',
    orient='records',
    force_ascii=False,
    indent=4
)
print(f"\nGotowe! Zapisano {len(df)} zdań.")