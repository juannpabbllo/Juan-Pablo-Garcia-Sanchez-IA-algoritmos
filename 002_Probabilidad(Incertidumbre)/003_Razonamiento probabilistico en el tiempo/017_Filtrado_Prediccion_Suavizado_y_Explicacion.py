import numpy as np

# Estados y observaciones
estados = ["S", "R"]
obs = ["CieloClaro", "Nublado"]

# Matriz de transición P(X_t | X_{t-1})
transicion = np.array([[0.7, 0.3],
                       [0.4, 0.6]])

# Probabilidades de observación P(e_t | X_t)
emision = np.array([[0.9, 0.1],  # CieloClaro dado Sol/R
                    [0.2, 0.8]]) # Nublado dado Sol/R

# Evidencia observada: e1 = CieloClaro, e2 = Nublado
evidencia = [0, 1]

# Distribución inicial
filtro = np.array([0.5, 0.5])

for e in evidencia:
    # Paso de predicción
    filtro = transicion.T @ filtro
    # Paso de corrección (filtrado)
    filtro = filtro * emision[e]
    filtro = filtro / np.sum(filtro)  # Normalizar

print(f"Distribución de estados después de la evidencia: {filtro}")
