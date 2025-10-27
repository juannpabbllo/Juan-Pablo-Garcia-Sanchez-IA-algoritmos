import numpy as np
import matplotlib.pyplot as plt

class AdalineGD:
    """
    Implementación de ADALINE (ADAptive LInear NEuron)
    con Descenso de Gradiente (Gradient Descent - GD).

    Parámetros
    ----------
    eta : float
        Tasa de aprendizaje (entre 0.0 y 1.0)
    n_iter : int
        Pasadas sobre el dataset de entrenamiento (épocas).

    Atributos
    ---------
    w_ : array-1d
        Pesos después del entrenamiento.
    b_ : escalar
        Sesgo (bias) después del entrenamiento.
    cost_ : list
        Suma de los errores al cuadrado (costo) en cada época.
    """
    
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Ajustar (entrenar) el modelo a los datos.
        
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples]
        """
        
        # Inicializamos pesos y sesgo (pequeños valores aleatorios o ceros)
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        self.cost_ = []

        for _ in range(self.n_iter):
            # 1. Calcular la entrada neta (z)
            net_input = self.net_input(X)
            
            # 2. Calcular el error (y_real - z)
            # Esta es la diferencia clave con el Perceptrón
            errors = y - net_input
            
            # 3. Actualizar pesos y sesgo (Regla Delta / Descenso de Gradiente)
            #    w = w + eta * sum(error * x_i)
            #    b = b + eta * sum(error)
            # (Usamos X.T.dot(errors) para la suma de todos los (error * x_i))
            self.w_ += self.eta * X.T.dot(errors)
            self.b_ += self.eta * errors.sum()
            
            # 4. Calcular el costo (Sum of Squared Errors - SSE)
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            
        print(f"Entrenamiento finalizado después de {self.n_iter} épocas.")
        print(f"Pesos finales: {self.w_}")
        print(f"Sesgo final: {self.b_}")
        return self

    def net_input(self, X):
        """Calcular la entrada neta (suma ponderada z)"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Devolver la etiqueta de clase (0 o 1) usando la función escalón"""
        # La predicción final SÍ usa la función escalón (step function)
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# --- Preparación de Datos (Compuerta Lógica AND) ---

# Entradas (X)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas deseadas (y)
y = np.array([0, 0, 0, 1])

# --- Entrenamiento del ADALINE ---

# 1. Crear una instancia de Adaline
#    (Usamos una tasa de aprendizaje 'eta' más pequeña que en el Perceptrón)
ada = AdalineGD(eta=0.01, n_iter=100)

# 2. Entrenar el modelo
ada.fit(X, y)

# --- Prueba del Modelo Entrenado ---
print("\n--- Probando el ADALINE entrenado ---")

print(f"0 AND 0 = {ada.predict([0, 0])}")
print(f"0 AND 1 = {ada.predict([0, 1])}")
print(f"1 AND 0 = {ada.predict([1, 0])}")
print(f"1 AND 1 = {ada.predict([1, 1])}")

# --- Visualización del Costo ---
# Esta es la parte más importante: ver cómo el error disminuye
print("\nGenerando gráfico de costo...")

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Épocas')
plt.ylabel('Suma de Errores al Cuadrado (Costo)')
plt.title('ADALINE - Convergencia del Descenso de Gradiente')
plt.grid(True)
plt.show()

print("Gráfico mostrado.")