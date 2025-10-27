# Ejemplo simple de una Red Bayesiana sin librerías externas

# Nodos: Fuego (F), Humo (H), Alarma (A)

# Probabilidades base
P_F = 0.05                          # P(F)
P_H_dado_F = 0.9                    # P(H|F)
P_H_dado_noF = 0.1                  # P(H|¬F)
P_A_dado_H = 0.95                   # P(A|H)
P_A_dado_noH = 0.05                 # P(A|¬H)

# Queremos calcular: P(F|A)
# Paso 1: Probabilidad total de que suene la alarma
P_H = (P_H_dado_F * P_F) + (P_H_dado_noF * (1 - P_F))
P_A = (P_A_dado_H * P_H) + (P_A_dado_noH * (1 - P_H))

# Paso 2: Aplicar Regla de Bayes: P(F|A) = P(A|F) * P(F) / P(A)
# Pero P(A|F) depende de H (intermedio)
P_A_dado_F = (P_A_dado_H * P_H_dado_F) + (P_A_dado_noH * (1 - P_H_dado_F))

P_F_dado_A = (P_A_dado_F * P_F) / P_A

print("=== Red Bayesiana ===")
print(f"P(Fuego | Alarma) = {P_F_dado_A:.3f}")
print(f"P(A) = {P_A:.3f}")
