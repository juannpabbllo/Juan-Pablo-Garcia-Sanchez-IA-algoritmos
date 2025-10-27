import numpy as np
import matplotlib.pyplot as plt

# --- 1. Funciones para crear Matrices de Transformación 3x3 ---
#    (Usando coordenadas homogéneas)

def create_translation_matrix(tx, ty):
    """Crea una matriz de traslación 2D."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def create_rotation_matrix(angle_degrees):
    """Crea una matriz de rotación 2D alrededor del origen."""
    angle_rad = np.radians(angle_degrees)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def create_scale_matrix(sx, sy):
    """Crea una matriz de escalado 2D."""
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

def plot_shape(points, style, label, **kwargs):
    """Función auxiliar para dibujar un polígono en matplotlib."""
    # points es (3, N)
    # Extraer x e y (ignorar w=1)
    x = points[0, :]
    y = points[1, :]
    
    # Cerrar el polígono (añadir el primer punto al final)
    x_plot = np.append(x, x[0])
    y_plot = np.append(y, y[0])
    
    plt.plot(x_plot, y_plot, style, label=label)

# --- 2. Definir el Objeto Original (un triángulo) ---
# Puntos como columnas:
# [x1, x2, x3]
# [y1, y2, y3]
# [ 1,  1,  1]
points_h = np.array([
    [0, 2, 1],  # Coordenadas X
    [0, 0, 2],  # Coordenadas Y
    [1, 1, 1]   # Coordenada W (homogénea)
])

# --- 3. Definir la Secuencia de Transformaciones ---
#
# El orden de multiplicación importa. Se aplica de derecha a izquierda:
# T @ R @ S @ Puntos
# 1. Escalar (S)
# 2. Rotar (R)
# 3. Trasladar (T)
#
print("Definiendo transformaciones:")
print("  1. Escalar 1.5x en X, 1.5x en Y")
S = create_scale_matrix(1.5, 1.5)

print("  2. Rotar 30 grados")
R = create_rotation_matrix(30)

print("  3. Trasladar 3 unidades en X, 1 unidad en Y")
T = create_translation_matrix(3, 1)

# --- 4. Aplicar las Transformaciones en Secuencia ---
print("\nAplicando transformaciones...")
# 1. Aplicar Escalado
points_scaled = S @ points_h

# 2. Aplicar Rotación (al resultado escalado)
points_rotated = R @ points_scaled

# 3. Aplicar Traslación (al resultado rotado y escalado)
points_final = T @ points_rotated

# También podemos hacerlo en un solo paso combinando las matrices
# M = T @ R @ S
# points_final_combined = M @ points_h

# --- 5. Visualizar los Resultados ---
print("Generando gráfico...")
plt.figure(figsize=(10, 8))
plt.title('Gráficos por Computador: Transformaciones 2D')

# Dibujar el original
plot_shape(points_h, 'b-', label='Original')

# Dibujar los pasos intermedios
plot_shape(points_scaled, 'g--', label='1. Escalado')
plot_shape(points_rotated, 'r--', label='2. Rotado')

# Dibujar el resultado final
plot_shape(points_final, 'k-', label='3. Trasladado (Final)', linewidth=2)

plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend()
plt.grid(True)
plt.axis('equal') # ¡Muy importante para que las rotaciones se vean correctas!
plt.show()

print("Gráfico mostrado.")