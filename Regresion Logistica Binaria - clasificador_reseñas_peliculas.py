"""Clasificación binaria de reseñas de películas con regresión logística.

Preprocesa los textos, los vectoriza con CountVectorizer, entrena el modelo, lo evalúa con el
conjunto de prueba y guarda el Pipeline entrenado (vectorizador + modelo) con joblib.

Uso:
    python clasificador_regresion_logistica.py <carpeta_con_train_y_test_json>
"""
import re
import sys
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

# Recursos de NLTK necesarios para tokenizar y para la lista de stop words
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Se conservan las negaciones: son relevantes en este contexto
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor"}
STEMMER = PorterStemmer()


def preprocesamiento(serie: pd.Series) -> pd.Series:
    """Minúsculas, limpieza de caracteres, tokenización, stop words y stemming."""
    serie = serie.str.lower()
    serie = serie.apply(lambda x: re.sub(r"[^a-z\s$]", "", x))
    serie = serie.apply(lambda x: word_tokenize(x, preserve_line=True))
    serie = serie.apply(lambda x: [STEMMER.stem(w) for w in x if w not in STOP_WORDS])
    return serie.apply(" ".join)


def main() -> None:
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

    # Entrenamiento
    df_train = pd.read_json(carpeta / "train.json")
    X_train = preprocesamiento(df_train["text"])
    y_train = df_train["label"]

    model = Pipeline([
        ("vectorizer", CountVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=1337)),
    ])
    model.fit(X_train, y_train)

    # Evaluación con el conjunto de prueba
    df_test = pd.read_json(carpeta / "test.json")
    X_test = preprocesamiento(df_test["text"])
    y_test = df_test["label"]
    predicciones = model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, predicciones))
    print("Precision:", precision_score(y_test, predicciones))
    print("Recall   :", recall_score(y_test, predicciones))
    print("F1-score :", f1_score(y_test, predicciones))

    joblib.dump(model, "clasificador_regresion_logistica.joblib")
    print("Modelo guardado en clasificador_regresion_logistica.joblib")


if __name__ == "__main__":
    main()
