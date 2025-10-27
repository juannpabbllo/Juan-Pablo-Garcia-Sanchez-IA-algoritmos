import numpy as np
import random

# Estados y acciones
states = ['Inicio', 'Medio', 'Meta']
actions = ['a', 'b']

# Recompensas por estado
R = {'Inicio':0, 'Medio':0, 'Meta':1}

# Probabilidades de transición (MDP)
T = {
    'Inicio': {'a': [('Medio', 1.0)], 'b': [('Inicio', 1.0)]},
    'Medio': {'a': [('Meta', 1.0)], 'b': [('Inicio', 1.0)]},
    'Meta': {'a': [('Meta', 1.0)], 'b': [('Meta', 1.0)]}
}

gamma = 0.9  # factor de descuento
alpha = 0.1  # tasa de aprendizaje
epsilon = 0.2  # exploración

# Inicializar Q-values
Q = {s: {a: 0 for a in actions} for s in states}

# Número de episodios
episodes = 500

for _ in range(episodes):
    state = 'Inicio'
    while state != 'Meta':
        # Epsilon-greedy: explorar o explotar
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = max(Q[state], key=Q[state].get)

        # Tomar acción y obtener siguiente estado y recompensa
        next_state = T[state][action][0][0]
        reward = R[next_state]

        # Actualizar Q-value
        Q[state][action] += alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])

        state = next_state

# Derivar política óptima
policy = {s: max(Q[s], key=Q[s].get) for s in states}

print("\n Q-values aprendidos:", Q)
print(" Política óptima aprendida:", policy)
