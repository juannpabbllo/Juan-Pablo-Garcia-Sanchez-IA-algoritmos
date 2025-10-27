import numpy as np
import matplotlib.pyplot as plt

# Definimos un rango de entradas (z)
# 100 puntos espaciados uniformemente entre -10 y 10
z = np.linspace(-10, 10, 100)

# --- Definición de las Funciones de Activación ---

def sigmoid(z):
    """Función Sigmoide: 1 / (1 + e^-z)"""
    return 1 / (1 + np.exp(-z))

def tanh(z):
    """Función Tangente Hiperbólica (integrada en numpy)"""
    return np.tanh(z)

def relu(z):
    """Función ReLU (Rectified Linear Unit): max(0, z)"""
    return np.maximum(0, z)

def step_function(z):
    """Función Escalón (como la del Perceptrón)"""
    return np.where(z >= 0, 1, 0)

# --- Cálculos y Salida en Texto ---
print("--- Probando las funciones con valores de ejemplo ---")

print(f"Valor de entrada (z) = -5.0")
print(f"  Sigmoid(-5.0) = {sigmoid(-5.0):.4f}")
print(f"  Tanh(-5.0)    = {tanh(-5.0):.4f}")
print(f"  ReLU(-5.0)    = {relu(-5.0):.4f}")

print(f"\nValor de entrada (z) = 0.0")
print(f"  Sigmoid(0.0)  = {sigmoid(0.0):.4f}")
print(f"  Tanh(0.0)     = {tanh(0.0):.4f}")
print(f"  ReLU(0.0)     = {relu(0.0):.4f}")

print(f"\nValor de entrada (z) = 5.0")
print(f"  Sigmoid(5.0)  = {sigmoid(5.0):.4f}")
print(f"  Tanh(5.0)     = {tanh(5.0):.4f}")
print(f"  ReLU(5.0)     = {relu(5.0):.4f}")

print("\nGenerando gráficos...")

# --- Visualización ---

# Aplicamos las funciones a nuestro rango 'z'
y_sigmoid = sigmoid(z)
y_tanh = tanh(z)
y_relu = relu(z)
y_step = step_function(z)

# Configurar la visualización (cuadrícula de 2x2)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Funciones de Activación Comunes', fontsize=16)

# Función Escalón (Perceptrón)
axs[0, 0].plot(z, y_step, label='Step (Escalón)', color='blue')
axs[0, 0].set_title('Función Escalón (Perceptrón)')
axs[0, 0].grid(True)
axs[0, 0].axhline(0, color='gray', lw=0.5)
axs[0, 0].axvline(0, color='gray', lw=0.5)

# Función Sigmoide
axs[0, 1].plot(z, y_sigmoid, label='Sigmoid', color='orange')
axs[0, 1].set_title('Función Sigmoide (Logística)')
axs[0, 1].grid(True)
axs[0, 1].axhline(0, color='gray', lw=0.5)
axs[0, 1].axvline(0, color='gray', lw=0.5)

# Función Tanh
axs[1, 0].plot(z, y_tanh, label='Tanh', color='green')
axs[1, 0].set_title('Función Tangente Hiperbólica (Tanh)')
axs[1, 0].grid(True)
axs[1, 0].axhline(0, color='gray', lw=0.5)
axs[1, 0].axvline(0, color='gray', lw=0.5)

# Función ReLU
axs[1, 1].plot(z, y_relu, label='ReLU', color='red')
axs[1, 1].set_title('Función ReLU')
axs[1, 1].grid(True)
axs[1, 1].axhline(0, color='gray', lw=0.5)
axs[1, 1].axvline(0, color='gray', lw=0.5)

# Ajustar diseño y mostrar
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Gráficos mostrados.")