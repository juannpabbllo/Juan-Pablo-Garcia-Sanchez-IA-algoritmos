import itertools

# --- 1. Definir el Diccionario de Vértices Válidos ---
#
# Este es el "cerebro" del algoritmo. Es un catálogo
# de todas las uniones de líneas FÍSICAMENTE POSIBLES
# en un mundo de bloques simple.
#
# Etiquetas:
#   '+' = Convexa
#   '-' = Cóncava
#   '->' = Límite (flecha en sentido horario)
#   '<-' = Límite (flecha en sentido antihorario)

# Para simplificar, solo usaremos '+' y '-' para bordes internos
# y 'B' para Límite (Boundary).

JUNCTION_DICTIONARY = {
    
    # --- TIPO 'L' (Dos líneas se encuentran) ---
    'L': [
        ('B', 'B'),  # Esquina de la silueta del objeto
        ('+', '-'),  # Borde convexo se encuentra con cóncavo
        ('+', '+'),  # Dos bordes convexos (raro, pero posible)
        ('-', '+')
        # (Y sus permutaciones)
    ],
    
    # --- TIPO 'FORK' o 'Y' (Tres líneas se encuentran) ---
    # (Vértice donde 3 caras se ven)
    'FORK': [
        ('+', '+', '+'),  # Esquina exterior de un cubo
        ('-', '-', '-')   # Esquina interior de una habitación
        # (No hay otras combinaciones posibles)
    ],
    
    # --- TIPO 'ARROW' o 'Flecha' (Tres líneas se encuentran) ---
    # (Vértice donde 2 caras se ven y una oculta)
    'ARROW': [
        ('+', 'B', 'B'),  # El palo de la flecha es convexo
        ('-', 'B', 'B')   # El palo de la flecha es cóncavo
        # (y sus rotaciones)
    ],
    
    # --- TIPO 'T' (Tres líneas se encuentran) ---
    # (Vértice donde un objeto está DELANTE de otro)
    'T': [
        ('B', 'B', '+'),  # El palo de la T es el objeto de atrás
        ('B', 'B', '-'),
        ('B', 'B', 'B')
        # La clave: las líneas de la T (horizontales) son
        # LÍMITES (B) del objeto que está delante.
    ]
}

# Añadir permutaciones al diccionario para que sea más fácil de buscar
def add_permutations(junction_list):
    full_list = set()
    for labels in junction_list:
        full_list.update(list(itertools.permutations(labels)))
    return list(full_list)

JUNCTION_DICTIONARY['L'] = add_permutations(JUNCTION_DICTIONARY['L'])
JUNCTION_DICTIONARY['FORK'] = add_permutations(JUNCTION_DICTIONARY['FORK'])
JUNCTION_DICTIONARY['ARROW'] = add_permutations(JUNCTION_DICTIONARY['ARROW'])
JUNCTION_DICTIONARY['T'] = add_permutations(JUNCTION_DICTIONARY['T'])

# --- 2. Función de Verificación ---

def check_labeling(junction_type, labels):
    """
    Verifica si un conjunto de etiquetas es válido para un
    tipo de vértice, consultando el diccionario.
    """
    if junction_type not in JUNCTION_DICTIONARY:
        return f"Tipo de vértice '{junction_type}' desconocido."
    
    if labels in JUNCTION_DICTIONARY[junction_type]:
        return "VÁLIDO"
    else:
        return "¡INVÁLIDO (Objeto Imposible)!"

# --- 3. Pruebas de Concepto ---
print("--- Probando el Etiquetado de Líneas (Mundo de Bloques) ---")

# Prueba 1: Una esquina de silueta simple (Tipo L)
labels_1 = ('B', 'B')
print(f"\nPrueba 1: Vértice tipo 'L', Etiquetas: {labels_1}")
print(f"  Resultado: {check_labeling('L', labels_1)}")

# Prueba 2: La esquina exterior de un cubo (Tipo Fork)
labels_2 = ('+', '+', '+')
print(f"\nPrueba 2: Vértice tipo 'FORK', Etiquetas: {labels_2}")
print(f"  Resultado: {check_labeling('FORK', labels_2)}")

# Prueba 3: La esquina interior de una habitación (Tipo Fork)
labels_3 = ('-', '-', '-')
print(f"\nPrueba 3: Vértice tipo 'FORK', Etiquetas: {labels_3}")
print(f"  Resultado: {check_labeling('FORK', labels_3)}")

# Prueba 4: Un vértice IMPOSIBLE (Tipo Fork)
# (Esto no puede existir en un objeto 3D real de caras planas)
labels_4 = ('+', '+', '-')
print(f"\nPrueba 4: Vértice tipo 'FORK', Etiquetas: {labels_4}")
print(f"  Resultado: {check_labeling('FORK', labels_4)}")

# Prueba 5: Oclusión (Tipo T)
# (Las dos líneas de la barra horizontal son límites, la vertical no)
labels_5 = ('B', 'B', '+')
print(f"\nPrueba 5: Vértice tipo 'T' (Oclusión), Etiquetas: {labels_5}")
print(f"  Resultado: {check_labeling('T', labels_5)}")