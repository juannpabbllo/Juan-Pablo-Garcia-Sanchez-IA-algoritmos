import sys
try:
    import numpy as np
except ImportError:
    print("Falta numpy. Instálelo con:\n    python -m pip install numpy")
    sys.exit(1)

try:
    import tensorflow as tf  # type: ignore[import]
    from tensorflow.keras.models import Sequential  # type: ignore[import]
    from tensorflow.keras.layers import Dense  # type: ignore[import]
    from tensorflow.keras.utils import to_categorical  # type: ignore[import]
except ImportError:
    print("Falta TensorFlow. Instálelo con:\n    python -m pip install tensorflow")
    sys.exit(1)

# Verificar versión de TensorFlow
print("TensorFlow version:", tf.__version__)

# Datos simulados (entrada: 4 características, 3 clases)
X = np.array([[0,0,0,0],
              [0,1,0,1],
              [1,0,1,0],
              [1,1,1,1]], dtype=np.float32)

y = np.array([0,1,2,0])

# Convertir etiquetas a one-hot
y_categ = to_categorical(y, num_classes=3)

# Crear modelo secuencial
model = Sequential()
model.add(Dense(5, input_dim=4, activation='relu'))  # capa oculta
model.add(Dense(3, activation='softmax'))            # capa salida

# Compilar modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar modelo
model.fit(X, y_categ, epochs=100, verbose=0)

# Predicción
nuevos = np.array([[0,1,1,0]], dtype=np.float32)  # asegurarse de que sea float32
predicciones = model.predict(nuevos)
print("Predicción (índice de clase):", np.argmax(predicciones))
print("Probabilidades:", predicciones)
