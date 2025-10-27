import numpy as np

# Estados ocultos: fonemas
estados = ["a", "e", "i", "o", "u"]

# Observaciones: caracteres reconocidos por sensor ruidoso
observaciones = ["a", "e", "i", "o", "u"]

# Matriz de transición simple (probabilidad de pasar de un fonema a otro)
transicion = np.full((5,5), 0.2)

# Probabilidades de observación (ruido)
emision = np.full((5,5), 0.2)
np.fill_diagonal(emision, 0.6)  # alta probabilidad de reconocer correctamente

# Evidencia observada (simulada)
evidencia = [0, 1, 2, 3, 4]  # "a", "e", "i", "o", "u"

# Forward-Backward simplificado
def forward_backward(transicion, emision, evidencia):
    T = len(evidencia)
    N = len(transicion)
    
    # Forward
    alpha = np.zeros((T,N))
    alpha[0,:] = emision[evidencia[0], :]
    for t in range(1,T):
        for j in range(N):
            alpha[t,j] = emision[evidencia[t], j] * np.sum(alpha[t-1,:] * transicion[:,j])
    
    # Backward
    beta = np.zeros((T,N))
    beta[T-1,:] = 1
    for t in reversed(range(T-1)):
        for i in range(N):
            beta[t,i] = np.sum(transicion[i,:] * emision[evidencia[t+1],:] * beta[t+1,:])
    
    # Suavizado
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma

gamma = forward_backward(transicion, emision, evidencia)

print("Probabilidad suavizada de cada fonema en cada tiempo:")
for t in range(len(evidencia)):
    print(f"t={t+1}: ", {estados[i]: round(gamma[t,i],3) for i in range(len(estados))})
