import random

def incertidumbre_evento(probabilidad_evento, repeticiones=1000):
    """
    Simula un evento incierto (como detectar fuego con un sensor)
    usando una probabilidad dada.
    """
    exitos = 0
    for _ in range(repeticiones):
        if random.random() < probabilidad_evento:
            exitos += 1
    frecuencia_observada = exitos / repeticiones
    return frecuencia_observada

# Ejemplo: probabilidad de que un sensor detecte fuego correctamente = 0.3
probabilidad_real = 0.3
resultado = incertidumbre_evento(probabilidad_real, 10000)

print(f"Probabilidad teórica: {probabilidad_real}")
print(f"Probabilidad observada tras la simulación: {resultado:.3f}")
