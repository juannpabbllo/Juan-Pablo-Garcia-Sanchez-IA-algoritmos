# Ejemplo: independencia condicional en un escenario de clima

# Probabilidades condicionadas al evento C = "Está lloviendo"
P_suelo_mojado_dado_lluvia = 0.9   # A | C
P_nubes_dado_lluvia = 0.95         # B | C

# Suponemos que las dos son independientes dado que llueve
P_conjunta_dado_lluvia = P_suelo_mojado_dado_lluvia * P_nubes_dado_lluvia

print("Independencia Condicional:")
print(f"P(Suelo mojado y Nubes | Lluvia) = {P_conjunta_dado_lluvia:.3f}")
print(f"P(Suelo mojado | Lluvia) = {P_suelo_mojado_dado_lluvia}")
print(f"P(Nubes | Lluvia) = {P_nubes_dado_lluvia}")
