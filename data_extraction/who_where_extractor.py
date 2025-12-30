INPUT = "data/bigger_categorise_data.json"
OUTPUT = "data/who-what.json"
sample_size = 300

import pandas as pd

# 1. Wczytanie danych
df = pd.read_json(INPUT)

# 2. Filtrowanie (usuwamy rekordy NO_EVENT)
df = df[df['category'] != 'NO_EVENT']

# 3. Losowanie 300 rekordów (lub mniej, jeśli po filtracji zostało ich mało)
df = df.sample(n=min(sample_size, len(df)))

# 4. Wybór kolumny 'sentence' i utworzenie kopii
df = df[['sentence', 'category']].copy()

# 5. Dodanie pustych kolumn (None w pandas zamieni się na null w JSON)
df['who'] = ""
df['trigger'] = ""
df['what'] = ""
df['where'] = ""
df['when'] = ""

# 6. Zapis do pliku
df.to_json(OUTPUT, orient='records', indent=4, force_ascii=False)