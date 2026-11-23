from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import numpy as np
import requests, json, os
from sklearn.neighbors import NearestNeighbors

DATASET_PATH = os.environ.get("DATASET_PATH", "assessment_dataset.npz")
OUTPUT_MODEL_PATH = os.environ.get("OUTPUT_MODEL_PATH", "poisoned_model_params.npz")
API_EVALUATOR_URL = os.environ.get("EVALUATOR_URL", "http://CAMBIAR_IP:PUERTO/evaluate_model")

arreglos = np.load(DATASET_PATH)
X_train = arreglos["X_train"]
y_train = arreglos["y_train"]
X_eval = arreglos["X_test"]
y_eval = arreglos["y_test"]

clasificador = LogisticRegression(random_state=1337, C=1.0, solver="liblinear", max_iter=200)
model = OneVsRestClassifier(clasificador)
model.fit(X_train, y_train)

predicciones = model.predict(X_eval)
print("metricas de evaluacion del modelo entrenado con el dataset sin envenenar")
print(classification_report(y_eval, predicciones))


def ataque_label_flipping_dirigido(y_train):
    y_train_envenenado = y_train.copy()
    porcentaje = float(input("Digite el porcentaje de las muestras de la clase 1 del dataset de entrenamiento que envenenara: "))
    indices_muestras_clase_1 = np.where(y_train == 1)[0]
    n_muestras_envenenar = int((len(indices_muestras_clase_1)) * porcentaje)
    generador = np.random.default_rng(1337)
    seleccionados = generador.choice(indices_muestras_clase_1, n_muestras_envenenar, replace=False)
    mitad = n_muestras_envenenar // 2
    indices_a_clase_2 = seleccionados[:mitad]
    indices_a_clase_0 = seleccionados[mitad:]

    y_train_envenenado[indices_a_clase_2] = 2
    y_train_envenenado[indices_a_clase_0] = 0

    return y_train_envenenado

y_train_envenenado = ataque_label_flipping_dirigido(y_train)
clasificador = LogisticRegression(random_state=1337, C=1.0, solver="liblinear", max_iter=200)
model_envenenado = OneVsRestClassifier(clasificador)
model_envenenado.fit(X_train, y_train_envenenado)

predicciones = model_envenenado.predict(X_eval)
print("\n\rmetricas de evaluacion del modelo entrenado con el dataset envenenado\n\r")
print(classification_report(y_eval, predicciones))

conteo = 0
params_to_save = {}
for i in model_envenenado.estimators_:
    params_to_save[f"coef_estimator_{conteo}"] = i.coef_
    params_to_save[f"intercept_estimator_{conteo}"] = i.intercept_
    conteo += 1
params_to_save["classes_"] = model_envenenado.classes_
np.savez_compressed(OUTPUT_MODEL_PATH, **params_to_save)

with open(OUTPUT_MODEL_PATH, "rb") as f:
    files = {"model_params": (OUTPUT_MODEL_PATH, f, "application/octet-stream")}
    response = requests.post(API_EVALUATOR_URL, files=files, timeout=30)

print(response.json())
