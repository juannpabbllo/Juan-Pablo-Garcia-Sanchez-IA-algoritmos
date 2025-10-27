import numpy as np

# Número de partículas
N = 1000

# Estado real inicial
x_real = 0.0

# Observaciones ruidosas
observaciones = [1.0, 2.0, 3.0, 2.5, 3.5]

# Inicializar partículas y pesos
particulas = np.random.normal(0, 1, N)
pesos = np.ones(N) / N

# Parámetros del modelo
Q = 0.1  # ruido de proceso
R = 0.5  # ruido de medición

estados_estimados = []

for z in observaciones:
    # Propagación de partículas
    particulas += np.random.normal(0, Q, N)
    
    # Actualización de pesos según la evidencia
    pesos = (1/np.sqrt(2*np.pi*R)) * np.exp(-0.5*((z-particulas)**2)/R)
    pesos /= np.sum(pesos)  # Normalizar
    
    # Estimación del estado
    x_est = np.sum(particulas * pesos)
    estados_estimados.append(x_est)
    
    # Re-muestreo
    indices = np.random.choice(np.arange(N), size=N, p=pesos)
    particulas = particulas[indices]
    pesos = np.ones(N) / N

print("Estados estimados con Filtrado de Partículas:")
print(estados_estimados)
