import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Datos simulados 2D
X = np.array([[1,2],[1,4],[1,0],
              [4,2],[4,4],[4,0]])

# Etiquetas para k-NN
y = np.array([0,0,0,1,1,1])

# --- k-NN (supervisado) ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
print("Predicción k-NN para [2,2]:", knn.predict([[2,2]]))

# --- k-Medias (no supervisado) ---
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("Centroides k-Medias:", kmeans.cluster_centers_)
print("Etiquetas k-Medias:", kmeans.labels_)
