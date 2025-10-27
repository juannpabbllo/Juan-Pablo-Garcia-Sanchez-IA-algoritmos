import numpy as np

# --- Función auxiliar para visualizar los patrones ---
def visualize_pattern(pattern, title=""):
    """Imprime un patrón 1D como una cuadrícula 2D (5x5)."""
    # Los patrones de Hopfield deben usar +1 y -1
    pattern_2d = pattern.reshape((5, 5))
    print(title)
    for row in pattern_2d:
        line = ""
        for pixel in row:
            if pixel == 1:
                line += "#"
            else:
                line += "."
        print(line)
    print("-" * 20)

class HopfieldNetwork:
    """Implementación de una Red de Hopfield."""
    
    def __init__(self, size):
        # La matriz de pesos será (size x size)
        self.size = size
        # Inicializamos la matriz de pesos en ceros
        self.weights = np.zeros((size, size))

    def store_pattern(self, pattern):
        """
        Almacenar un patrón usando la Regla de Hebb.
        El patrón debe ser bipolar (-1, 1).
        """
        # 1. Regla de Hebb:
        #    La actualización de pesos es el producto exterior del patrón consigo mismo.
        #    W = W + (pattern.T * pattern)
        update = np.outer(pattern, pattern)
        
        # 2. Restricción: las neuronas no se conectan a sí mismas
        #    Ponemos la diagonal de la matriz de pesos a cero.
        np.fill_diagonal(update, 0)
        
        self.weights += update
        print(f"Almacenando patrón... Pesos actualizados.")

    def store_patterns(self, patterns):
        """Almacena una lista de patrones bipolares (-1, 1)."""
        for pattern in patterns:
            self.store_pattern(pattern)
        print("\nAlmacenamiento de patrones completado.")

    def retrieve(self, noisy_pattern, max_iter=10):
        """
        Recuperar un patrón a partir de una entrada ruidosa.
        Utiliza actualización síncrona.
        """
        current_pattern = np.copy(noisy_pattern)
        
        for i in range(max_iter):
            print(f"Paso de recuperación {i+1}:")
            visualize_pattern(current_pattern)
            
            # 1. Calcular la "energía" o activación neta
            #    net_input = W * current_pattern
            net_input = np.dot(self.weights, current_pattern)
            
            # 2. Función de activación (Sign/Step)
            #    Si net_input >= 0, la neurona se activa (+1)
            #    Si net_input < 0, la neurona se desactiva (-1)
            new_pattern = np.sign(net_input)
            
            # 3. Corregir ceros (np.sign(0) = 0, lo queremos como +1 o -1)
            #    Si la entrada neta es 0, mantenemos el estado anterior
            new_pattern[new_pattern == 0] = current_pattern[new_pattern == 0]
            
            # 4. Comprobar si la red se ha estabilizado
            if np.array_equal(current_pattern, new_pattern):
                print(f"La red se ha estabilizado en el paso {i+1}.")
                break
            
            current_pattern = new_pattern
            
        return current_pattern

# --- 1. Definición de Patrones (5x5 = 25 neuronas) ---
# Los definimos con 0 y 1 para que sean fáciles de leer
pattern_C_01 = np.array([
    [1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0]
])

pattern_T_01 = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

pattern_L_01 = np.array([
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]
])

# --- 2. Convertir Patrones a Bipolar (-1, 1) ---
# Las redes de Hopfield funcionan con -1 y +1
def to_bipolar(pattern):
    return (pattern * 2) - 1

patterns_bipolar = [
    to_bipolar(pattern_C_01.flatten()),
    to_bipolar(pattern_T_01.flatten()),
    to_bipolar(pattern_L_01.flatten())
]

# --- 3. Crear y Almacenar Patrones en la Red ---
network_size = 25
hopfield_net = HopfieldNetwork(size=network_size)
hopfield_net.store_patterns(patterns_bipolar)

visualize_pattern(patterns_bipolar[1], title="Patrón 'T' original almacenado:")

# --- 4. Crear Patrón Ruidoso ---
noisy_T_01 = np.array([
    [1, 1, 0, 1, 1],  # error aquí
    [0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0],  # error aquí
    [0, 0, 1, 0, 1],  # error aquí
    [0, 0, 1, 0, 0]
])
noisy_T_bipolar = to_bipolar(noisy_T_01.flatten())

visualize_pattern(noisy_T_bipolar, title="Patrón 'T' ruidoso (Entrada):")

# --- 5. Recuperar el Patrón ---
retrieved_pattern = hopfield_net.retrieve(noisy_T_bipolar, max_iter=10)

visualize_pattern(retrieved_pattern, title="Patrón 'T' recuperado (Salida):")