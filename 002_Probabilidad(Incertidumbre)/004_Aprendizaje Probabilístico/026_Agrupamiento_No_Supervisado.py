import numpy as np

# Datos simulados (2D)
datos = np.array([[1,2],[1,4],[1,0],
                  [4,2],[4,4],[4,0]])

# Número de clusters
k = 2

# Inicializar centroides al azar
centroides = datos[np.random.choice(datos.shape[0], k, replace=False)]

for _ in range(10):  # número de iteraciones
    # Asignar puntos al cluster más cercano
    distancias = np.linalg.norm(datos[:, np.newaxis] - centroides, axis=2)
    etiquetas = np.argmin(distancias, axis=1)
    
    # Actualizar centroides
    for i in range(k):
        centroides[i] = datos[etiquetas == i].mean(axis=0)

print("Centroides finales:", centroides)
print("Etiquetas de cada punto:", etiquetas)

