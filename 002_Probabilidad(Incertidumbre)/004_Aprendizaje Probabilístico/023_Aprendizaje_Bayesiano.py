# Probabilidades a priori
P_llovera = 0.3
P_no_llovera = 0.7

# Verosimilitudes
P_nublado_dado_llovera = 0.8
P_nublado_dado_no_llovera = 0.4

# Evidencia total P(D)
P_nublado = P_nublado_dado_llovera * P_llovera + P_nublado_dado_no_llovera * P_no_llovera

# Probabilidad posterior
P_llovera_dado_nublado = (P_nublado_dado_llovera * P_llovera) / P_nublado

print(f"Probabilidad de que llueva dado que está nublado: {P_llovera_dado_nublado:.3f}")
