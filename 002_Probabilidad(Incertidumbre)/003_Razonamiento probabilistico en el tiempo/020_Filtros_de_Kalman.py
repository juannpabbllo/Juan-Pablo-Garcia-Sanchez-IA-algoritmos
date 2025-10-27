import numpy as np

# Inicialización
x = 0.0        # Estado inicial
P = 1.0        # Incertidumbre inicial

A = 1.0        # Transición de estado
H = 1.0        # Observación directa
Q = 0.01       # Varianza del proceso
R = 0.1        # Varianza de la medición

# Observaciones ruidosas
mediciones = [1.0, 2.0, 3.0, 2.5, 3.5]

estados_estimados = []

for z in mediciones:
    # Predicción
    x_pred = A * x
    P_pred = A * P * A + Q
    
    # Actualización (corrección)
    K = P_pred * H / (H * P_pred * H + R)  # Ganancia de Kalman
    x = x_pred + K * (z - H * x_pred)
    P = (1 - K * H) * P_pred
    
    estados_estimados.append(x)

print("Estados estimados con Kalman:")
print(estados_estimados)
