import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom # Importamos la librería

# --- 1. Preparación de Datos ---
# Vamos a crear 9 colores puros (R,G,B)
# Cada color es un vector de 3 dimensiones

# Datos de entrada (9 muestras, 3 características)
data = np.array([
    [1, 0, 0], # Rojo
    [0, 1, 0], # Verde
    [0, 0, 1], # Azul
    [1, 1, 0], # Amarillo
    [0, 1, 1], # Cian
    [1, 0, 1], # Magenta
    [1, 1, 1], # Blanco
    [0, 0, 0], # Negro
    [0.5, 0.5, 0.5] # Gris
])

# Nombres de los colores para el gráfico
color_names = ['Rojo', 'Verde', 'Azul', 'Amarillo', 'Cian', 'Magenta', 'Blanco', 'Negro', 'Gris']

# --- 2. Configuración e Inicialización del SOM ---

# Tamaño del mapa (cuadrícula de 10x10 neuronas)
map_x = 10
map_y = 10

# input_len = 3 porque nuestros datos son (R, G, B)
# sigma = radio inicial del vecindario
# learning_rate = tasa de aprendizaje inicial
som = MiniSom(x=map_x, y=map_y, input_len=3, 
            sigma=1.0, learning_rate=0.5,
            random_seed=42)

# Inicializamos los pesos del mapa con valores aleatorios
som.random_weights_init(data)

print("Iniciando entrenamiento del SOM...")
print(f"Mapa: {map_x}x{map_y} neuronas")
print(f"Datos: {len(data)} muestras (colores)")

# --- 3. Entrenamiento de la Red ---
# Entrenamos el mapa por 1000 iteraciones
som.train_random(data=data, num_iteration=1000)

print("Entrenamiento finalizado.")

# --- 4. Visualización del Mapa de Pesos ---
print("Generando gráfico 1: El mapa de colores organizado...")

# Obtenemos los pesos finales del mapa.
# Esto es una cuadrícula de 10x10, y cada celda tiene un vector (R,G,B)
# Shape: (10, 10, 3)
map_colors = som.get_weights()

# Mostramos esta cuadrícula como una imagen
plt.figure(figsize=(7, 7))
plt.title('Mapa de Colores Autoorganizado (10x10)')
# 'imshow' toma los datos (10,10,3) y los interpreta como una imagen
plt.imshow(map_colors)
plt.xticks([])
plt.yticks([])

# --- 5. Visualización de los Nodos Ganadores (BMU) ---
print("Generando gráfico 2: Ubicación de los datos de entrada...")

# Este gráfico mostrará en qué parte del mapa "aterrizó"
# cada uno de nuestros 9 colores de entrada.

# Usamos un fondo blanco para el mapa
plt.figure(figsize=(7, 7))
plt.title('Ubicación de los Colores de Entrada en el Mapa')

# 'pcolor' crea la cuadrícula. Usamos los pesos como fondo
# (aquí usamos 'bone_r' que es un mapa de blanco y negro)
plt.pcolor(map_colors[:,:,0], cmap='bone_r') # Usamos solo el canal Rojo como fondo
plt.colorbar(label='Intensidad (solo como referencia)')

# Encontramos la BMU (la neurona ganadora) para cada uno de nuestros 9 colores
# y dibujamos un círculo de ese color en su ubicación.
for i, (color_vec, color_name) in enumerate(zip(data, color_names)):
    # som.winner(x) devuelve la coordenada (x,y) de la BMU
    (w_x, w_y) = som.winner(color_vec)
    
    # Ponemos el marcador en el centro de la celda (+0.5)
    # Usamos 'color=color_vec' para que el marcador tenga el color de entrada
    plt.plot(w_x + 0.5, w_y + 0.5, 
             'o', # forma de círculo
             markersize=15, 
             markerfacecolor=color_vec, 
             markeredgecolor='k', # borde negro
             label=color_name)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()

print("Gráficos mostrados.")