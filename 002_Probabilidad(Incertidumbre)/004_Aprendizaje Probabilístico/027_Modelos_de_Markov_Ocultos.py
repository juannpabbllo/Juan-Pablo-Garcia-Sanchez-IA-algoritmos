import numpy as np

# Estados ocultos
estados = ["Lluvia", "Soleado"]
# Observaciones
observaciones = ["Caminar", "Comprar", "Limpiar"]
O_seq = ["Caminar", "Comprar", "Limpiar"]

# Índices para matrices
obs_idx = {o:i for i,o in enumerate(observaciones)}
state_idx = {s:i for i,s in enumerate(estados)}

# Matriz de transición
A = np.array([[0.7, 0.3],
              [0.4, 0.6]])

# Matriz de emisión
B = np.array([[0.1, 0.4, 0.5],
              [0.6, 0.3, 0.1]])

# Probabilidades iniciales
pi = np.array([0.6, 0.4])

# Algoritmo Viterbi
T = len(O_seq)
N = len(estados)
delta = np.zeros((T,N))
psi = np.zeros((T,N), dtype=int)

# Inicialización
delta[0,:] = pi * B[:, obs_idx[O_seq[0]]]

# Recursión
for t in range(1, T):
    for j in range(N):
        seq_probs = delta[t-1,:] * A[:, j]
        psi[t,j] = np.argmax(seq_probs)
        delta[t,j] = np.max(seq_probs) * B[j, obs_idx[O_seq[t]]]

# Decodificación de la secuencia más probable
estados_ocultos = np.zeros(T, dtype=int)
estados_ocultos[T-1] = np.argmax(delta[T-1,:])
for t in reversed(range(1,T)):
    estados_ocultos[t-1] = psi[t, estados_ocultos[t]]

# Mostrar resultados
secuencia_estimada = [estados[i] for i in estados_ocultos]
print("Secuencia de estados ocultos más probable:", secuencia_estimada)
