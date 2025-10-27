import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- Copiamos la clase Perceptron de la Práctica 031 ---
# (No hay cambios en la clase en sí)
class Perceptron:
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.eta * (target - prediction)
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        
        print(f"Entrenamiento finalizado después de {self.n_iter} épocas.")
        print(f"Pesos finales: {self.w_}")
        print(f"Sesgo final: {self.b_}")
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# --- Función Auxiliar para Graficar ---
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 'x', 's', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
    
    plt.xlabel('Característica 1 (x1)')
    plt.ylabel('Característica 2 (x2)')
    plt.legend(loc='upper left')
    plt.grid(True)


# --- Preparación de Datos (Compuerta Lógica XOR) ---

# Entradas (X): 4 muestras, 2 características
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas deseadas (y): ¡EL PROBLEMA XOR!
y_xor = np.array([0, 1, 1, 0])

# --- Entrenamiento del Perceptrón en XOR ---

# 1. Crear una instancia del Perceptrón
#    (Aumentamos las iteraciones a 100 para darle más oportunidad de aprender)
ppn_xor = Perceptron(eta=0.1, n_iter=100)

# 2. Entrenar el modelo con los datos XOR
ppn_xor.fit(X, y_xor)

# --- Prueba del Modelo Entrenado ---
print("\n--- Probando el Perceptrón con XOR ---")
print(f"0 XOR 0 = {ppn_xor.predict([0, 0])}")
print(f"0 XOR 1 = {ppn_xor.predict([0, 1])}")
print(f"1 XOR 0 = {ppn_xor.predict([1, 0])}")
print(f"1 XOR 1 = {ppn_xor.predict([1, 1])}")

# --- Visualización del Fracaso ---
print("\nGenerando gráfico de errores y frontera de decisión...")

# Gráfico 1: Errores por época
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(ppn_xor.errors_) + 1), ppn_xor.errors_, marker='o')
plt.xlabel('Épocas')
plt.ylabel('Número de errores')
plt.title('Errores de entrenamiento (XOR)')
plt.grid(True)

# Gráfico 2: Frontera de decisión
plt.subplot(1, 2, 2)
plot_decision_regions(X, y_xor, classifier=ppn_xor)
plt.title('Frontera de Decisión (XOR)')

plt.tight_layout()
plt.show()

print("Gráficos mostrados.")