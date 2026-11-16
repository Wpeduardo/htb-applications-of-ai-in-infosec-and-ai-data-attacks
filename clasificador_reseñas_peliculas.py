"""Clasificación binaria de reseñas de peliculas con Naive Bayes Multinomial.

Preprocesa los textos, busca el mejor valor de alpha con GridSearchCV, evalúa el modelo con el
conjunto de prueba y lo guarda con joblib.

Uso:
    python clasificador_sentimiento.py <carpeta_con_train_y_test_json>
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Recursos de NLTK necesarios para tokenizar y para la lista de stop words
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Se conservan las negaciones: son relevantes para el sentimiento
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor"}
STEMMER = PorterStemmer()


def preprocesamiento(serie: pd.Series) -> pd.Series:
    """Minúsculas, limpieza de caracteres, tokenización, stop words y stemming."""
    serie = serie.str.lower()
    serie = serie.apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
    serie = serie.apply(word_tokenize)
    serie = serie.apply(lambda x: [STEMMER.stem(w) for w in x if w not in STOP_WORDS])
    return serie.apply(" ".join)


def main() -> None:
    carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

    # Entrenamiento
    df_train = pd.read_json(carpeta / "train.json")
    X_train = preprocesamiento(df_train["text"])
    y_train = df_train["label"]

    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))),
        ("classifier", MultinomialNB()),
    ])
    param_grid = {"classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]}

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Mejor alpha:", grid_search.best_params_["classifier__alpha"])

    # Evaluación con el conjunto de prueba
    df_test = pd.read_json(carpeta / "test.json")
    X_test = preprocesamiento(df_test["text"])
    y_test = df_test["label"]
    predicciones = best_model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, predicciones))
    print("Precision:", precision_score(y_test, predicciones))
    print("Recall   :", recall_score(y_test, predicciones))
    print("F1-score :", f1_score(y_test, predicciones))

    # Guardar el Pipeline entrenado (vectorizador + clasificador)
    joblib.dump(best_model, "clasificador.joblib")
    print("Modelo guardado en clasificador.joblib")


if __name__ == "__main__":
    main()
