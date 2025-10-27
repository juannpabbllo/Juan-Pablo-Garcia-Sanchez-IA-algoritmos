import numpy as np

# Estados y observaciones
estados = ["S", "R"]
obs = ["C", "N"]

# Matriz de transición P(X_t | X_{t-1})
transicion = np.array([[0.7, 0.3],
                       [0.4, 0.6]])

# Probabilidades de observación P(e_t | X_t)
emision = np.array([[0.9, 0.1],  # C dado S/R
                    [0.2, 0.8]]) # N dado S/R

# Evidencia observada: C=0, N=1, C=0
evidencia = [0, 1, 0]

# Distribución inicial
pi = np.array([0.5, 0.5])

# Forward
T = len(evidencia)
N = len(estados)
alpha = np.zeros((T, N))
alpha[0, :] = pi * emision[evidencia[0], :]
for t in range(1, T):
    for j in range(N):
        alpha[t, j] = emision[evidencia[t], j] * np.sum(alpha[t-1, :] * transicion[:, j])

# Backward
beta = np.zeros((T, N))
beta[T-1, :] = 1
for t in reversed(range(T-1)):
    for i in range(N):
        beta[t, i] = np.sum(transicion[i, :] * emision[evidencia[t+1], :] * beta[t+1, :])

# Suavizado: P(X_t | e_1:T)
gamma = alpha * beta
gamma /= gamma.sum(axis=1, keepdims=True)

print("Distribución de estados en cada tiempo:")
for t in range(T):
    print(f"t={t+1}: Sol={gamma[t,0]:.3f}, Lluvia={gamma[t,1]:.3f}")
