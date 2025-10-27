import numpy as np

# Re-utilizamos la función Sigmoide de la práctica 032
# Es necesaria porque su "suavidad" (ser derivable)
# es lo que permite el entrenamiento (que veremos después).
def sigmoid(z):
    """Función Sigmoide: 1 / (1 + e^-z)"""
    return 1 / (1 + np.exp(-z))

# --- Definición de la Arquitectura y Pesos Manuales para XOR ---
#
# Nuestra red será:
# - 2 neuronas de entrada (x1, x2)
# - 1 capa oculta con 2 neuronas (h1, h2)
# - 1 capa de salida con 1 neurona (o1)

# --- Pesos de la Capa Oculta (Entrada -> Oculta) ---
# w_h[i, j] = peso desde entrada 'i' a neurona oculta 'j'
# Vamos a "enseñarle" a h1 a ser OR y a h2 a ser NAND
w_h = np.array([
    [20, -20],  # Pesos de x1 (a h1 y h2)
    [20, -20]   # Pesos de x2 (a h1 y h2)
])

# Sesgos (bias) de la Capa Oculta (uno para h1, uno para h2)
# b_h[j] = sesgo para la neurona oculta 'j'
b_h = np.array([-10, 30])  # Sesgo para OR (h1) y NAND (h2)

# --- Pesos de la Capa de Salida (Oculta -> Salida) ---
# w_o[j, k] = peso desde neurona oculta 'j' a neurona de salida 'k'
# Vamos a "enseñarle" a la salida (o1) a ser AND
w_o = np.array([
    [20],  # Peso de h1 (OR) a la salida
    [20]   # Peso de h2 (NAND) a la salida
])

# Sesgo (bias) de la Capa de Salida
b_o = np.array([-30]) # Sesgo para AND

# --- Función de Predicción (Propagación hacia Adelante) ---
def predict(X):
    """
    Realiza una pasada hacia adelante (forward pass) a través de la red.
    X debe ser un array de entrada, ej: [1, 0]
    """
    
    # 1. Calcular activación de la capa oculta (h)
    #    z_h = (X · w_h) + b_h
    z_h = np.dot(X, w_h) + b_h
    a_h = sigmoid(z_h)
    
    # 2. Calcular activación de la capa de salida (o)
    #    z_o = (a_h · w_o) + b_o
    z_o = np.dot(a_h, w_o) + b_o
    a_o = sigmoid(z_o)
    
    # 3. Devolver la predicción
    #    Si la salida es > 0.5, es 1. Si no, 0.
    prediction = 1 if a_o > 0.5 else 0
    
    # Imprimimos los valores intermedios para ver la "magia"
    print(f"  Entrada: {X}")
    print(f"  Capa Oculta (OR, NAND): {np.round(a_h, 2)}")
    print(f"  Capa de Salida (AND): {a_o[0]:.4f} -> Predicción: {prediction}")
    
    return prediction

# --- Preparación de Datos (Compuerta Lógica XOR) ---

# Entradas (X)
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas deseadas (y)
y_xor = np.array([0, 1, 1, 0])

# --- Probando la Red Multicapa ---
print("--- Probando la Red Multicapa (MLP) con XOR ---")
print("(Pesos definidos manualmente)\n")

for i in range(len(X_xor)):
    xi = X_xor[i]
    yi = y_xor[i]
    
    print(f"Calculando {xi[0]} XOR {xi[1]}:")
    pred = predict(xi)
    print(f"  Resultado esperado: {yi} -> {'¡CORRECTO!' if pred == yi else '¡INCORRECTO!'}")
    print("-" * 30)