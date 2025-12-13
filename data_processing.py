import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_name, binary = False, only_events = False):
    # file_name = 'data/result.json'

    df = pd.read_json(file_name)

    encoder = None
    if binary:
        y = (df['category'] != "NO_EVENT").astype(int)
    else:
        if only_events:
            df = df[df["category"] != "NO_EVENT"]
        encoder = LabelEncoder()
        encoder.fit(df['category']) 
        y = encoder.transform(df['category'])

    print(f"Wczytano {len(df)} rekordów.")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['sentence'], y, test_size=0.2, random_state=42, stratify=y
    )

    return (X_train_raw, y_train), (X_test_raw, y_test), encoder

def bag_of_words(X_train_raw, X_test_raw):
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    return X_train, X_test

def SBERT_mini(X_train_raw, X_test_raw, cache_file='data/embeddings_cache_mini.npz'):
    if os.path.exists(cache_file):
        print(f"Znaleziono cache: {cache_file}. Ładowanie...")
        data = np.load(cache_file)
        if len(data['X_train']) == len(X_train_raw):
            return data['X_train'], data['X_test']
        else:
            print("Uwaga: Rozmiar danych w cache nie zgadza się z surowymi danymi. Przeliczam ponownie.")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generowanie embeddingów treningowych...")
    X_train = model.encode(X_train_raw.tolist(), show_progress_bar=True)
    
    print("Generowanie embeddingów testowych...")
    X_test = model.encode(X_test_raw.tolist(), show_progress_bar=True)

    print(f"Wymiar macierzy cech X: {X_train.shape}")
    np.savez_compressed(cache_file, X_train=X_train, X_test=X_test)
    
    return X_train, X_test

def SBERT(X_train_raw, X_test_raw, cache_file='data/embeddings_cache.npz'):
    if os.path.exists(cache_file):
        print(f"Znaleziono cache: {cache_file}. Ładowanie...")
        data = np.load(cache_file)
        if len(data['X_train']) == len(X_train_raw):
            return data['X_train'], data['X_test']
        else:
            print("Uwaga: Rozmiar danych w cache nie zgadza się z surowymi danymi. Przeliczam ponownie.")

    model = SentenceTransformer('all-mpnet-base-v2')

    print("Generowanie embeddingów treningowych...")
    X_train = model.encode(X_train_raw.tolist(), show_progress_bar=True)
    
    print("Generowanie embeddingów testowych...")
    X_test = model.encode(X_test_raw.tolist(), show_progress_bar=True)

    print(f"Wymiar macierzy cech X: {X_train.shape}")
    np.savez_compressed(cache_file, X_train=X_train, X_test=X_test)
    
    return X_train, X_test