import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import feature  # <--- De scikit-image

def load_image(file_name):
    """
    Función auxiliar para cargar la imagen, manejar errores
    y convertir a escala de grises.
    """
    print(f"Cargando '{file_name}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, file_name)
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: No se pudo cargar la imagen.")
        print(f"Asegúrate de que '{file_name}' exista en la ruta:")
        print(f"{image_path}")
        return None
    
    # Convertir a escala de grises para el análisis de textura
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def calculate_lbp(image):
    """
    Calcula la imagen LBP y su histograma.
    """
    print("Calculando Patrones Binarios Locales (LBP)...")
    
    # P = 8 puntos (vecinos)
    # R = 1 (radio de 1 píxel)
    # method="uniform" es una variante de LBP más robusta
    lbp = feature.local_binary_pattern(image, P=8, R=1, method="uniform")
    
    # Calcular el histograma de los códigos LBP
    # (El número de bins es 10 para el método "uniform" con P=8)
    n_bins = int(lbp.max() + 1)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=n_bins,
                             range=(0, n_bins))
    
    # Normalizar el histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6) # (1e-6 para evitar división por cero)
    
    return lbp, hist

# --- 1. Cargar Imágenes de Textura ---
# (Asegúrate de tener estos archivos en la carpeta)
img_a = load_image('textura_A.jpg') # Ej: Madera
img_b = load_image('textura_b.jpg') # Ej: Tela

if img_a is None or img_b is None:
    print("No se pudieron cargar una o ambas imágenes. Saliendo.")
    exit()

# --- 2. Calcular LBP y Histogramas ---
lbp_a, hist_a = calculate_lbp(img_a)
lbp_b, hist_b = calculate_lbp(img_b)

# --- 3. Mostrar Resultados ---
print("Mostrando resultados...")

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Análisis de Textura con Local Binary Patterns (LBP)', fontsize=16)

# --- Fila 1: Textura A ---
axs[0, 0].imshow(img_a, cmap='gray')
axs[0, 0].set_title('Original (Textura A)')
axs[0, 0].axis('off')

axs[0, 1].imshow(lbp_a, cmap='gray')
axs[0, 1].set_title('Imagen LBP (Textura A)')
axs[0, 1].axis('off')

axs[0, 2].plot(hist_a)
axs[0, 2].set_title('Histograma LBP (Textura A)')
axs[0, 2].set_ylim(0, 0.4) # Fijar eje Y para comparar

# --- Fila 2: Textura B ---
axs[1, 0].imshow(img_b, cmap='gray')
axs[1, 0].set_title('Original (Textura B)')
axs[1, 0].axis('off')

axs[1, 1].imshow(lbp_b, cmap='gray')
axs[1, 1].set_title('Imagen LBP (Textura B)')
axs[1, 1].axis('off')

axs[1, 2].plot(hist_b)
axs[1, 2].set_title('Histograma LBP (Textura B)')
axs[1, 2].set_ylim(0, 0.4) # Fijar eje Y para comparar

plt.tight_layout()
plt.show()

print("Proceso completado.")