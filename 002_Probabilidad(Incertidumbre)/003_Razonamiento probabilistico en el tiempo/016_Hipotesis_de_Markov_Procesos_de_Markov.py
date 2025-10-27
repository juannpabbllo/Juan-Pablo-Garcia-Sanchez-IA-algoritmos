import random

# Estados y matriz de transición
estados = ["S", "N", "L"]
transicion = {
    "S": [0.6, 0.3, 0.1],
    "N": [0.3, 0.4, 0.3],
    "L": [0.2, 0.3, 0.5]
}

def simular_markov(inicio="S", pasos=10):
    actual = inicio
    historial = [actual]
    for _ in range(pasos-1):
        actual = random.choices(estados, weights=transicion[actual])[0]
        historial.append(actual)
    return historial

historial_clima = simular_markov()
print("Simulación de clima (Cadena de Markov):")
print(historial_clima)
