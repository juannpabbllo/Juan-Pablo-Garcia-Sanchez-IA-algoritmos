# Ejemplo: Un robot detecta humo y quiere saber la probabilidad de que haya fuego
# Dado que el sensor de humo puede fallar.

# Probabilidades base
P_fuego = 0.05                 # Probabilidad a priori de fuego
P_sin_fuego = 1 - P_fuego

# Sensibilidad del sensor (verdadero positivo y falso positivo)
P_s_humo_dado_fuego = 0.9      # P(Sensor detecta humo | hay fuego)
P_s_humo_dado_no_fuego = 0.2   # P(Sensor detecta humo | no hay fuego)

# Probabilidad total de que el sensor detecte humo
P_humo = (P_s_humo_dado_fuego * P_fuego) + (P_s_humo_dado_no_fuego * P_sin_fuego)

# Probabilidad condicionada de fuego dado que se detecta humo (Regla de Bayes)
P_fuego_dado_humo = (P_s_humo_dado_fuego * P_fuego) / P_humo

# Normalización implícita (la suma de ambas debe ser 1)
P_no_fuego_dado_humo = 1 - P_fuego_dado_humo

print("Probabilidad Condicionada (Regla de Bayes):")
print(f"P(Fuego | Humo) = {P_fuego_dado_humo:.3f}")
print(f"P(No Fuego | Humo) = {P_no_fuego_dado_humo:.3f}")
