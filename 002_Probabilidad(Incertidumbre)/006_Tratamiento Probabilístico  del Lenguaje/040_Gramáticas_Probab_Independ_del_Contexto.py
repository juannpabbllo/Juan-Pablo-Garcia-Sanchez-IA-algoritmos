import nltk
from nltk import PCFG
from nltk.parse import ViterbiParser

# --- 1. Definición de la Gramática Probabilística (PCFG) ---
#
# S = Oración (Sentence)
# NP = Sintagma Nominal (Noun Phrase, ej: "el hombre", "yo")
# VP = Sintagma Verbal (Verb Phrase, ej: "vi al hombre")
# PP = Sintagma Preposicional (Prepositional Phrase, ej: "con el telescopio")
# Det = Determinante (ej: "the")
# N = Sustantivo (Noun, ej: "man", "telescope")
# P = Preposición (ej: "with")
# V = Verbo (ej: "saw")
#
# Las probabilidades [x.y] indican la probabilidad de cada regla.

# Esta es la gramática que define la AMBIGÜEDAD:
# 1. VP -> V NP PP [0.4] (El PP modifica al VP -> "Vi [usando] el telescopio")
# 2. NP -> NP PP   [0.3] (El PP modifica al NP -> "El hombre [que tiene] el telescopio")
# Le damos más probabilidad a la regla (2) para que el parser la prefiera.

grammar_string = """
    S -> NP VP  [1.0]

    NP -> 'I'       [0.1]
    NP -> Det N     [0.3]
    NP -> NP PP     [0.6]  # <-- Regla 2 (Modifica al hombre)

    VP -> V NP      [0.4]
    VP -> V NP PP   [0.6]  # <-- Regla 1 (Modifica al verbo)
    
    PP -> P NP      [1.0]
    
    Det -> 'the'    [1.0]
    P -> 'with'     [1.0]
    V -> 'saw'      [1.0]
    N -> 'man'      [0.5]
    N -> 'telescope'[0.5]
"""

# --- 2. Cargar la Gramática y el Parser ---

# Crear el objeto de gramática PCFG
try:
    pcfg_grammar = PCFG.fromstring(grammar_string)
    print("Gramática PCFG cargada correctamente.")
except ValueError as e:
    print(f"Error al cargar la gramática: {e}")
    exit()

# Crear un ViterbiParser (un algoritmo eficiente para encontrar el mejor árbol)
parser = ViterbiParser(pcfg_grammar)

# --- 3. Analizar (Parse) la Frase ---

# La frase ambigua, separada en tokens (palabras)
sentence = "I saw the man with the telescope".split()

print(f"\nAnalizando la frase: '{' '.join(sentence)}'")

# Parsear la frase. 'parser.parse()' devuelve un iterador
# con los árboles posibles, ordenados del más probable al menos probable.
parses = list(parser.parse(sentence))

# --- 4. Mostrar Resultados ---

if not parses:
    print("¡Error! La frase no se puede analizar con esta gramática.")
else:
    print(f"\nSe encontraron {len(parses)} posibles análisis.")
    print("Mostrando el árbol de análisis (parse tree) MÁS PROBABLE:")
    
    # El primer árbol en la lista es el más probable
    best_parse = parses[0]
    
    # .prob() nos da la probabilidad total del árbol
    # (el producto de todas las probabilidades de las reglas usadas)
    print(f"\nProbabilidad Total del Árbol: {best_parse.prob():.10f}")
    
    # Imprimir el árbol de forma bonita
    best_parse.pretty_print()
    
    # También podemos dibujarlo en una ventana nueva (opcional)
    # print("\n(Cierra la ventana del árbol para continuar...)")
    # best_parse.draw()