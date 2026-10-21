"""Neurona de McCulloch-Pitts para clasificar tumores (1 = maligno, 0 = benigno)."""
import numpy as np


class NeuronaMP:
    def __init__(self):
        self.umbrales = None            # un umbral por cada característica
        self.theta = None               # votos necesarios para declarar "maligno"
        self.exactitud_por_theta = {}   # para poder mostrar cómo se eligió theta

    def binarizar(self, X):
        """Cada valor pasa a 1 si supera el umbral de su columna; si no, a 0."""
        return (np.asarray(X) > self.umbrales).astype(int)

    def sumar(self, X):
        """Cuenta cuántas características 'votan' 1 en cada fila."""
        return self.binarizar(X).sum(axis=1)

    def entrenar(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # 1) Umbral de cada entrada = mediana de su columna
        self.umbrales = np.median(X, axis=0)

        # 2) Probar todos los theta posibles y guardar la exactitud de cada uno
        sumas = self.sumar(X)
        self.exactitud_por_theta = {}
        for theta in range(X.shape[1] + 1):
            prediccion = (sumas >= theta).astype(int)
            self.exactitud_por_theta[theta] = float((prediccion == y).mean())

        # 3) Quedarse con el theta que más acierta
        self.theta = max(self.exactitud_por_theta, key=self.exactitud_por_theta.get)
        return self

    def predict(self, X):
        """Devuelve 1 (maligno) o 0 (benigno) por cada fila."""
        return (self.sumar(X) >= self.theta).astype(int)
