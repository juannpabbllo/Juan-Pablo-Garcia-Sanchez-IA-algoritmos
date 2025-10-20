import numpy as np
import math

# Ejemplo MDP: 3 estados y 2 acciones
states = [0, 1, 2]  # 0: inicio, 2: meta
actions = ['a', 'b']

# Recompensas por estado (se toma la recompensa del estado resultante)
R = [0, 0, 1]  # solo la meta tiene recompensa 1

# Probabilidades de transición (T[s][a] = lista de (prob, next_state))
T = {
    0: {
        'a': [(0.8, 1), (0.2, 0)],
        'b': [(1.0, 0)]
    },
    1: {
        'a': [(1.0, 2)],
        'b': [(1.0, 0)]
    },
    2: {
        'a': [(1.0, 2)],
        'b': [(1.0, 2)]
    }
}

gamma = 0.9  # factor de descuento
theta = 1e-5  # tolerancia para convergencia
max_iterations = 10000  # seguridad para evitar bucles infinitos

def value_iteration(states, actions, T, R, gamma, theta, max_iter=10000):
    # Inicializar valores
    V = [0.0 for _ in states]

    for it in range(max_iter):
        delta = 0.0
        for s in states:
            v = V[s]
            # calcular el valor por acción
            action_values = []
            for a in actions:
                q = 0.0
                for p, s_ in T[s][a]:
                    # recompensa tomada como función del siguiente estado
                    q += p * (R[s_] + gamma * V[s_])
                action_values.append(q)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    else:
        # llegó al máximo de iteraciones
        pass

    # Extraer política óptima
    policy = {}
    for s in states:
        best_action = None
        best_value = -math.inf
        for a in actions:
            q = sum(p * (R[s_] + gamma * V[s_]) for p, s_ in T[s][a])
            if q > best_value:
                best_value = q
                best_action = a
        policy[s] = best_action

    return V, policy

if __name__ == "__main__":
    V, policy = value_iteration(states, actions, T, R, gamma, theta, max_iterations)
    print("Valores de los estados:")
    for s, v in enumerate(V):
        print(f"  Estado {s}: {v:.6f}")
    print("Política óptima:", policy)
