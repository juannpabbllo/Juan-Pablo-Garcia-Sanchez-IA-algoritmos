import numpy as np

# --- 1. Definir la Frase Ambigua ---
sentence = "I saw the man with the telescope"

# --- 2. Definir las Probabilidades de ESTRUCTURA (de la PCFG anterior) ---

# Probabilidad de la estructura 1: "with the telescope" modifica a "saw"
# (VP -> V NP PP)
p_structure_1 = 0.4 

# Probabilidad de la estructura 2: "with the telescope" modifica a "man"
# (NP -> NP PP)
p_structure_2 = 0.6

print("--- Simulación de una PCFG Simple (Práctica 040) ---")
print(f"Prob. Estructura 1 (Ver con telescopio): {p_structure_1}")
print(f"Prob. Estructura 2 (Hombre con telescopio): {p_structure_2}")
if p_structure_1 > p_structure_2:
    print("-> Ganador PCFG: Estructura 1 (Ver con telescopio)\n")
else:
    print("-> Ganador PCFG: Estructura 2 (Hombre con telescopio)\n")


# --- 3. Definir las Probabilidades LÉXICAS (¡NUEVO!) ---
#
# Estas probabilidades las aprendería un modelo L-PCFG
# a partir de un corpus gigante (Treebank).
# Representan la "afinidad" entre palabras.

# Probabilidad de que el PP(with) modifique al VP(saw)
# P(PP(with) | VP(saw))
# "Ver con" es una construcción posible (ver con mis ojos),
# pero "ver con un telescopio" es menos común.
p_lexical_1 = 0.05

# Probabilidad de que el PP(with) modifique al NP(man)
# P(PP(with) | NP(man))
# "Un hombre con [algo]" es una construcción muy común
# (hombre con sombrero, hombre con perro, hombre con telescopio).
p_lexical_2 = 0.30

print("--- Simulación de una L-PCFG (Esta Práctica) ---")
print("Añadiendo probabilidades léxicas (dependientes de palabras):")
print(f"  Prob. Léxica 1 (P('with' | 'saw')): {p_lexical_1}")
print(f"  Prob. Léxica 2 (P('with' | 'man')): {p_lexical_2}\n")


# --- 4. Calcular el Ganador L-PCFG ---
#
# La probabilidad total del árbol es el producto de todas
# las probabilidades (estructura y léxicas).

# P(Árbol 1) = P(Estructura 1) * P(Léxica 1)
total_prob_1 = p_structure_1 * p_lexical_1

# P(Árbol 2) = P(Estructura 2) * P(Léxica 2)
total_prob_2 = p_structure_2 * p_lexical_2

print("--- Resultados Finales (L-PCFG) ---")
print(f"Prob. Total Árbol 1 (Ver con telescopio): {p_structure_1} * {p_lexical_1} = {total_prob_1:.4f}")
print(f"Prob. Total Árbol 2 (Hombre con telescopio): {p_structure_2} * {p_lexical_2} = {total_prob_2:.4f}")

if total_prob_1 > total_prob_2:
    print("\n-> Ganador L-PCFG: Árbol 1 (Ver con telescopio)")
    print("   El modelo cree que 'yo' usé el telescopio para ver.")
else:
    print("\n-> Ganador L-PCFG: Árbol 2 (Hombre con telescopio)")
    print("   El modelo cree que el 'hombre' tenía el telescopio.")