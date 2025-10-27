import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Clase MLP con Retropropagación ---

class MLP_Backpropagation:
    """
    Implementación de un Perceptrón Multicapa (MLP)
    entrenado con Retropropagación (Backpropagation).
    
    Arquitectura fija: Entrada(2) -> Oculta(2) -> Salida(1)
    """
    
    def __init__(self, eta=0.1, n_iter=10000):
        self.eta = eta          # Tasa de aprendizaje
        self.n_iter = n_iter    # Épocas de entrenamiento
        
        # 1. Inicialización de pesos y sesgos
        # Pesos (aleatorios) y sesgos (ceros)
        # (Usamos una semilla para que el resultado sea reproducible)
        np.random.seed(42)
        
        # Pesos: Entrada(2) -> Oculta(2)
        self.w_h = np.random.uniform(-1, 1, (2, 2))
        self.b_h = np.zeros(2)
        
        # Pesos: Oculta(2) -> Salida(1)
        self.w_o = np.random.uniform(-1, 1, (2, 1))
        self.b_o = np.zeros(1)
        
        self.cost_ = []

    def _sigmoid(self, z):
        # Clip para evitar 'overflow' con números muy grandes en exp()
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _sigmoid_derivative(self, a):
        # a = sigmoid(z)
        # La derivada de la sigmoide es a * (1 - a)
        return a * (1 - a)

    def _forward(self, X):
        """Paso 1: Propagación hacia adelante"""
        # Propagación: Entrada -> Oculta
        z_h = np.dot(X, self.w_h) + self.b_h
        a_h = self._sigmoid(z_h) # Activación capa oculta
        
        # Propagación: Oculta -> Salida
        z_o = np.dot(a_h, self.w_o) + self.b_o
        a_o = self._sigmoid(z_o) # Activación capa salida
        
        # Devolvemos las activaciones, las necesitaremos para el backward pass
        return z_h, a_h, z_o, a_o

    def fit(self, X, y):
        """Entrenar la red."""
        
        # 'y' debe tener la forma [n_samples, n_output]
        y_reshaped = y.reshape(-1, 1)
        
        print(f"Iniciando entrenamiento con Backpropagation...")
        
        for epoch in range(self.n_iter):
            # Iteramos sobre cada muestra de entrenamiento (xi, target)
            # Esto se llama Descenso de Gradiente Estocástico (SGD)
            for xi, target in zip(X, y_reshaped):
                
                # --- 1. Forward Pass ---
                z_h, a_h, z_o, a_o = self._forward(xi)
                
                # --- 2. Backward Pass (Calcular Errores y Gradientes) ---
                
                # Error en la capa de salida
                # (target - a_o) * derivada_sigmoide(z_o)
                delta_o = (target - a_o) * self._sigmoid_derivative(a_o)
                
                # Error en la capa oculta (propagado hacia atrás)
                # (delta_o · w_o.T) * derivada_sigmoide(z_h)
                delta_h = np.dot(delta_o, self.w_o.T) * self._sigmoid_derivative(a_h)
                
                # --- 3. Calcular Gradientes (Δw) ---
                #    (Cuánto cambiar cada peso)
                
                # Gradiente: Oculta -> Salida
                grad_w_o = np.outer(a_h, delta_o)
                grad_b_o = delta_o
                
                # Gradiente: Entrada -> Oculta
                grad_w_h = np.outer(xi, delta_h)
                grad_b_h = delta_h

                # --- 4. Actualizar Pesos y Sesgos ---
                self.w_o += self.eta * grad_w_o
                self.b_o += self.eta * grad_b_o
                
                self.w_h += self.eta * grad_w_h
                self.b_h += self.eta * grad_b_h

            # --- Fin del bucle de muestras ---
            
            # Calcular el costo total de la época
            _, _, _, a_o_total = self._forward(X)
            cost = np.mean((y_reshaped - a_o_total)**2) / 2
            self.cost_.append(cost)
            
            if (epoch + 1) % 1000 == 0:
                print(f'  Época: {epoch + 1}/{self.n_iter}, Costo: {cost:.6f}')
        
        print("Entrenamiento de Retropropagación finalizado.")
        return self

    def predict(self, X):
        """Realizar predicciones finales (después de entrenar)."""
        # Hacemos un forward pass
        _, _, _, a_o = self._forward(X)
        # Redondear al más cercano (0 o 1)
        return np.round(a_o)

# --- Función Auxiliar para Graficar (de la Práctica 034) ---
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 'x', 's', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.ravel() # Aplanar a 1D
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    plt.grid(True)


# --- Preparación de Datos (XOR) ---
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0]) # Salidas deseadas

# --- Configuración y Entrenamiento de la Red ---
# Aumentamos las iteraciones (épocas) y la tasa de aprendizaje
# para asegurarnos de que aprenda bien el XOR
mlp = MLP_Backpropagation(eta=0.1, n_iter=10000)

mlp.fit(X, y)

# --- Prueba del Modelo Entrenado ---
print("\n--- Probando la Red (MLP) entrenada con XOR ---")

# (Aplanamos la salida de .predict() para que sea más legible)
print(f"0 XOR 0 = {mlp.predict([0, 0])[0]:.0f}  (Esperado: 0)")
print(f"0 XOR 1 = {mlp.predict([0, 1])[0]:.0f}  (Esperado: 1)")
print(f"1 XOR 0 = {mlp.predict([1, 0])[0]:.0f}  (Esperado: 1)")
print(f"1 XOR 1 = {mlp.predict([1, 1])[0]:.0f}  (Esperado: 0)")

# --- Visualización del Aprendizaje ---
print("\nGenerando gráficos de aprendizaje...")

plt.figure(figsize=(10, 5))

# Gráfico 1: Costo (Error) vs. Época
plt.subplot(1, 2, 1)
plt.plot(range(1, len(mlp.cost_) + 1), mlp.cost_)
plt.xlabel('Épocas')
plt.ylabel('Costo (Error Cuadrático Medio)')
plt.title('Convergencia de Backpropagation')
plt.grid(True)
plt.tight_layout()

# Gráfico 2: Frontera de decisión (¡la prueba visual!)
plt.subplot(1, 2, 2)
plot_decision_regions(X, y, classifier=mlp)
plt.title('Frontera de Decisión (XOR Resuelto)')
plt.xlabel('Característica 1 (x1)')
plt.ylabel('Característica 2 (x2)')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

print("Gráficos mostrados.")