import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

print("Cargando 'imagen.webp'...")

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'imagen.webp')

if not os.path.exists(image_path):
    print(f"Error: el archivo no existe en: {image_path}")
    sys.exit(1)

# Intentar con OpenCV primero
img = cv2.imread(image_path)
if img is None:
    print("cv2.imread devolvió None. Intentando fallback con Pillow (PIL)...")
    try:
        from PIL import Image
    except ImportError:
        print("Falta Pillow. Instálelo con:\n    python -m pip install pillow")
        sys.exit(1)
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        print("Carga con Pillow correcta.")
    except Exception as e:
        print("Error al cargar con Pillow:", e)
        sys.exit(1)
else:
    print("Imagen cargada correctamente con OpenCV.")

# --- De aquí en adelante, el código es el mismo ---

# Convertir a escala de grises
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 2. Detección de Aristas (Canny) ---
print("Aplicando Detección de Aristas (Canny)...")
canny_edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)


# --- 3. Segmentación (Otsu's Thresholding) ---
print("Aplicando Segmentación (Otsu)...")

blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

threshold_value, segmented_image = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print(f"El umbral óptimo de Otsu se encontró en: {threshold_value}")

# --- 4. Mostrar Resultados ---
print("Mostrando resultados...")

fig, axs = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Detección de Aristas vs. Segmentación', fontsize=16)

axs[0].imshow(img_gray, cmap='gray')
axs[0].set_title('Original (Escala de Grises)')
axs[0].axis('off')

axs[1].imshow(canny_edges, cmap='gray')
axs[1].set_title('Detección de Aristas (Canny)')
axs[1].axis('off') # (Corregí un error aquí, decía axs[0] dos veces)

axs[2].imshow(segmented_image, cmap='gray')
axs[2].set_title('Segmentación (Otsu)')
axs[2].axis('off')

plt.tight_layout()
plt.show()

print("Proceso completado.")