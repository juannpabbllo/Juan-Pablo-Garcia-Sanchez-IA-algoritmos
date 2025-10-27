# Ejemplo: Aplicación de la Regla de Bayes
# Hipótesis: hay fuego
# Evidencia: sensor detecta humo

# Probabilidades base
P_fuego = 0.05                    # Probabilidad a priori de fuego
P_sin_fuego = 1 - P_fuego
P_humo_dado_fuego = 0.9           # Sensor detecta humo si hay fuego
P_humo_dado_no_fuego = 0.2        # Sensor detecta humo por error

# Probabilidad total de la evidencia (humo)
P_humo = (P_humo_dado_fuego * P_fuego) + (P_humo_dado_no_fuego * P_sin_fuego)

# Aplicación de la Regla de Bayes
P_fuego_dado_humo = (P_humo_dado_fuego * P_fuego) / P_humo
P_no_fuego_dado_humo = 1 - P_fuego_dado_humo

print("=== Regla de Bayes ===")
print(f"P(Fuego | Humo) = {P_fuego_dado_humo:.3f}")
print(f"P(No Fuego | Humo) = {P_no_fuego_dado_humo:.3f}")
