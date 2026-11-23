## Ataque: Label Flipping dirigido (clasificación ambigua de la Clase 1)

**Objetivo:** envenenar el dataset de entrenamiento mediante *label flipping* para que el
modelo clasifique de forma ambigua las muestras de la **Clase 1**, repartiendo sus
predicciones entre la **Clase 0** y la **Clase 2**, sin degradar de forma notable el
rendimiento sobre el resto de clases.

**Estrategia:** se toma un porcentaje de las muestras de la Clase 1 en el conjunto de
entrenamiento y se cambian sus etiquetas: la mitad a Clase 0 y la mitad a Clase 2.
Esto "parte" la región de la Clase 1 por ambos lados y obliga al optimizador a desplazar
las fronteras de decisión, reduciendo el *recall* de esa clase.

Modelo: `OneVsRestClassifier(LogisticRegression(C=1.0, solver="liblinear", max_iter=200))`
Porcentaje envenenado: **40 %** de las muestras de la Clase 1.

### Resultados

**Modelo base (dataset limpio)** — accuracy global: **0.93**

| Clase | Precision | Recall | F1-score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| 0 | 0.92 | 0.99 | 0.95 | 125 |
| 1 | 0.96 | **0.74** | 0.83 | 125 |
| 2 | 0.90 | 0.98 | 0.94 | 125 |
| 3 | 0.93 | 0.99 | 0.96 | 125 |

**Modelo envenenado (40 %)** — accuracy global: **0.78**

| Clase | Precision | Recall | F1-score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| 0 | 0.74 | 1.00 | 0.85 | 125 |
| 1 | 1.00 | **0.12** | 0.21 | 125 |
| 2 | 0.68 | 0.99 | 0.81 | 125 |
| 3 | 0.93 | 0.99 | 0.96 | 125 |

### Análisis

- El **recall de la Clase 1 cae de 0.74 a 0.12**: tras el ataque, el modelo solo reconoce
  correctamente el 12 % de las muestras reales de esa clase. Las demás se desvían hacia las
  Clases 0 y 2, que es la "ambigüedad" buscada.
- La **precisión de la Clase 1 sube a 1.00** porque el modelo se vuelve muy conservador:
  casi nunca predice Clase 1, y cuando lo hace acierta. El daño se mide en el *recall*, no en
  la precisión.
- Las **precisiones de las Clases 0 y 2 bajan** (0.92 → 0.74 y 0.90 → 0.68) porque ahora
  reciben muestras que en realidad eran de Clase 1.
- La **Clase 3 no se ve afectada** (recall 0.99), confirmando que el ataque es dirigido y no
  degrada el modelo de forma global indiscriminada.
