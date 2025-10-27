import numpy as np

# Estados y observaciones
estados = ["S", "R"]
obs = ["C", "N"]

# Matriz de transición P(X_t | X_{t-1})
transicion = np.array([[0.7, 0.3],
                       [0.4, 0.6]])

# Probabilidades de observación P(E_t | X_t)
emision = np.array([[0.9, 0.1],
                    [0.2, 0.8]])

# Distribución inicial
pi = np.array([0.5, 0.5])

# Evidencia observada
evidencia = [0, 1, 0]

# Función Forward (filtrado)
def forward(pi, transicion, emision, evidencia):
    T = len(evidencia)
    N = len(pi)
    alpha = np.zeros((T, N))
    alpha[0, :] = pi * emision[evidencia[0], :]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = emision[evidencia[t], j] * np.sum(alpha[t-1, :] * transicion[:, j])
    return alpha

# Función Backward
def backward(transicion, emision, evidencia):
    T = len(evidencia)
    N = transicion.shape[0]
    beta = np.zeros((T, N))
    beta[T-1, :] = 1
    for t in reversed(range(T-1)):
        for i in range(N):
            beta[t, i] = np.sum(transicion[i, :] * emision[evidencia[t+1], :] * beta[t+1, :])
    return beta

# Forward-Backward (suavizado)
alpha = forward(pi, transicion, emision, evidencia)
beta = backward(transicion, emision, evidencia)
gamma = alpha * beta
gamma /= gamma.sum(axis=1, keepdims=True)

print("Distribución de estados (suavizado) en cada tiempo:")
for t in range(len(evidencia)):
    print(f"t={t+1}: Sol={gamma[t,0]:.3f}, Lluvia={gamma[t,1]:.3f}")

