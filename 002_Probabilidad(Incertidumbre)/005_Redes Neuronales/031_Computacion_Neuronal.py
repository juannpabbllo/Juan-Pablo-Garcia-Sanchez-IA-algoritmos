import numpy as np

class Perceptron:
    
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        
        # Inicializamos pesos y sesgo con ceros
        # n_features es el número de columnas de X (en nuestro caso, 2)
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            # Iteramos sobre cada muestra de entrenamiento (xi, target)
            for xi, target in zip(X, y):
                # 1. Calcular la salida (predicción)
                #    La función de activación (escalón) está en el paso 2
                prediction = self.predict(xi)
                
                # 2. Calcular el error y actualizar pesos
                #    update = eta * (y_real - y_predicha)
                update = self.eta * (target - prediction)
                
                # 3. Actualizar pesos y sesgo
                #    w_j = w_j + update * xi_j
                #    b   = b   + update
                self.w_ += update * xi
                self.b_ += update
                
                # Contar si hubo un error
                errors += int(update != 0.0)
                
            self.errors_.append(errors)
            # print(f"Época {_+1}: Pesos {self.w_}, Sesgo {self.b_}, Errores {errors}")
        
        print(f"Entrenamiento finalizado después de {self.n_iter} épocas.")
        print(f"Pesos finales: {self.w_}")
        print(f"Sesgo final: {self.b_}")
        return self

    def net_input(self, X):
        """Calcular la entrada neta (suma ponderada)"""
        # z = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Devolver la etiqueta de clase (0 o 1) usando la función escalón"""
        # Es la función de activación
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# --- Preparación de Datos (Compuerta Lógica AND) ---

# Entradas (X): 4 muestras, 2 características cada una
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas deseadas (y): 4 etiquetas
y = np.array([0, 0, 0, 1])

# --- Entrenamiento del Perceptrón ---

# 1. Crear una instancia del Perceptrón
#    (tasa de aprendizaje de 0.1, 10 épocas de entrenamiento)
ppn = Perceptron(eta=0.1, n_iter=10)

# 2. Entrenar el modelo con los datos AND
ppn.fit(X, y)

# --- Prueba del Modelo Entrenado ---
print("\n--- Probando el Perceptrón entrenado ---")

print(f"0 AND 0 = {ppn.predict([0, 0])}")
print(f"0 AND 1 = {ppn.predict([0, 1])}")
print(f"1 AND 0 = {ppn.predict([1, 0])}")
print(f"1 AND 1 = {ppn.predict([1, 1])}")

# Opcional: ver cómo disminuyeron los errores por época
# import matplotlib.pyplot as plt
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Épocas')
# plt.ylabel('Número de errores')
# plt.title('Errores de entrenamiento del Perceptrón')
# plt.grid(True)
# plt.show()