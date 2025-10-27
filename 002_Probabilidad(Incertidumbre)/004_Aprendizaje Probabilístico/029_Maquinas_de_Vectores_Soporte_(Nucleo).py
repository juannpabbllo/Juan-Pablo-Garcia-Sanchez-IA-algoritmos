import numpy as np
from sklearn import svm

# Datos simulados (2D)
X = np.array([[1,2],[2,3],[3,3],
              [6,5],[7,7],[8,6]])
y = np.array([0,0,0,1,1,1])  # clases

# Crear clasificador SVM lineal
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Predicción para nuevos puntos
nuevos = np.array([[3,2],[5,5]])
predicciones = clf.predict(nuevos)

print("Predicciones SVM para nuevos puntos:", predicciones)
print("Vectores de soporte:", clf.support_vectors_)
