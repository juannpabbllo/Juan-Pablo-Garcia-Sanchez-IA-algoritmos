import numpy as np

# Datos simulados (alturas)
datos = np.array([1.60, 1.65, 1.70, 1.72, 1.75, 1.80, 1.82, 1.85])

# Inicializar parámetros de las 2 gaussianas
mu = np.array([1.65, 1.80])      # medias
sigma = np.array([0.05, 0.05])   # desviaciones
pi = np.array([0.5, 0.5])        # pesos de la mezcla

# Número de iteraciones
n_iter = 10

for _ in range(n_iter):
    # E-step: calcular responsabilidades
    resp = np.zeros((len(datos), 2))
    for k in range(2):
        resp[:, k] = pi[k] * (1/np.sqrt(2*np.pi*sigma[k]**2)) * np.exp(-(datos - mu[k])**2 / (2*sigma[k]**2))
    resp /= resp.sum(axis=1, keepdims=True)
    
    # M-step: actualizar parámetros
    for k in range(2):
        Nk = resp[:, k].sum()
        mu[k] = (resp[:, k] @ datos) / Nk
        sigma[k] = np.sqrt((resp[:, k] @ (datos - mu[k])**2) / Nk)
        pi[k] = Nk / len(datos)

print("Parámetros estimados:")
print("Medias:", mu)
print("Desviaciones:", sigma)
print("Pesos de mezcla:", pi)
