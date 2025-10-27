import random

# Distribución estacionaria de tráfico
probabilidades = {"Alto": 0.3, "Medio": 0.5, "Bajo": 0.2}

def simular_trafico(tiempos=10):
    historial = []
    for _ in range(tiempos):
        # Generar tráfico según distribución fija
        estado = random.choices(list(probabilidades.keys()), weights=list(probabilidades.values()))[0]
        historial.append(estado)
    return historial

historial_trafico = simular_trafico()
print("Simulación de tráfico (Proceso Estacionario):")
print(historial_trafico)
