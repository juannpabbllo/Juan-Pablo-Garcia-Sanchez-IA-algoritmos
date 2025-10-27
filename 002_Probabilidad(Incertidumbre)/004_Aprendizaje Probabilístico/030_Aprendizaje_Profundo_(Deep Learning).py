import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Datos simulados (entrada: 4 características, 3 clases)
X = np.array([[0,0,0,0],
              [0,1,0,1],
              [1,0,1,0],
              [1,1,1,1]])
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
nuevos = np.array([[0,1,1,0]])
predicciones = model.predict(nuevos)
print("Predicción:", np.argmax(predicciones))