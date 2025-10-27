# Ejemplo de la Regla de la Cadena
P_A = 0.3      # Probabilidad de que llueva
P_B_given_A = 0.9   # Probabilidad de llevar paraguas dado que llueve
P_C_given_A_B = 0.1 # Probabilidad de mojarse dado que llueve y llevas paraguas

# Aplicar regla de la cadena
P_A_B_C = P_A * P_B_given_A * P_C_given_A_B

print(f"La probabilidad conjunta P(A,B,C) = {P_A_B_C:.3f}")
