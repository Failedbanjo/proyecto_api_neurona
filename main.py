"""Servidor FastAPI: entrena la neurona al arrancar y muestra un formulario."""
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from neurona_mp import NeuronaMP

# Las 10 caracteristicas elegidas (el orden importa)
CARACTERISTICAS = [
    "worst concave points",
    "worst perimeter",
    "worst radius",
    "worst area",
    "mean concavity",
    "mean concave points",
    "worst texture",
    "mean perimeter",
    "mean area",
    "worst concavity",
]

# ---- 1) Entrenar la neurona ----
datos = load_breast_cancer()
X = pd.DataFrame(datos.data, columns=datos.feature_names)[CARACTERISTICAS]
y = 1 - datos.target  # 1 = maligno, 0 = benigno

X_ent, X_prueba, y_ent, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
neurona = NeuronaMP().entrenar(X_ent.values, y_ent)
print(f"Neurona lista -> theta = {neurona.theta}")

# ---- 2) Crear la app ----
app = FastAPI(title="Clasificador de tumores con neurona MP")


class Muestra(BaseModel):
    """Pydantic exige 10 numeros (no negativos; area, radio, perimetro y textura mayores que 0)."""
    worst_concave_points: float = Field(ge=0)
    worst_perimeter: float = Field(gt=0)
    worst_radius: float = Field(gt=0)
    worst_area: float = Field(gt=0)
    mean_concavity: float = Field(ge=0)
    mean_concave_points: float = Field(ge=0)
    worst_texture: float = Field(gt=0)
    mean_perimeter: float = Field(gt=0)
    mean_area: float = Field(gt=0)
    worst_concavity: float = Field(ge=0)


# ---- 3) Ruta GET para el formulario
@app.get("/", response_class=HTMLResponse)
def pagina_principal():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


# ---- 4) Ruta POST para los 10 valores y devuelve el diagnostico ----
@app.post("/predict")
def predecir(m: Muestra):
    # Se arma la fila en el mismo orden con el que se entreno la neurona
    fila = [[getattr(m, c.replace(" ", "_")) for c in CARACTERISTICAS]]
    votos = int(neurona.sumar(fila)[0])
    es_maligno = int(neurona.predict(fila)[0]) == 1
    return {
        "diagnostico": "Maligno" if es_maligno else "Benigno",
        "votos": votos,
        "theta": int(neurona.theta),
    }
