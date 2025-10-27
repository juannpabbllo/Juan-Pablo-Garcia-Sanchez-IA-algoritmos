import numpy as np

# Datos de entrenamiento (simplificado)
# Cada fila: [Humedad, Viento], Clase
datos = [
    ["Alta", "Fuerte", "Sí"],
    ["Alta", "Calmo", "Sí"],
    ["Baja", "Fuerte", "No"],
    ["Baja", "Calmo", "No"],
]

# Nueva instancia a predecir
instancia = ["Alta", "Fuerte"]

# Probabilidades a priori P(C)
clases = ["Sí", "No"]
P_C = {c: sum(1 for fila in datos if fila[2]==c)/len(datos) for c in clases}

# Probabilidades condicionales P(X_i | C)
P_X_given_C = {}
for c in clases:
    subset = [fila for fila in datos if fila[2]==c]
    for i in range(2):  # dos características
        valores = [fila[i] for fila in subset]
        for valor in set(valores):
            P_X_given_C[(i, valor, c)] = valores.count(valor)/len(subset)

# Clasificación
posteriors = {}
for c in clases:
    posterior = P_C[c]
    for i, val in enumerate(instancia):
        posterior *= P_X_given_C.get((i, val, c), 0.01)  # suavizado
    posteriors[c] = posterior

# Normalizar
total = sum(posteriors.values())
for c in posteriors:
    posteriors[c] /= total

print("Probabilidades de cada clase:", posteriors)
