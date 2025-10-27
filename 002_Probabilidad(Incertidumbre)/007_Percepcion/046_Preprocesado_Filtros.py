import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def add_salt_and_pepper(image, amount):
    """
    Añade ruido de "sal y pimienta" a una imagen en escala de grises.
    amount: Proporción de píxeles a afectar (ej: 0.05 = 5%)
    """
    noisy_img = image.copy()
    num_pixels = image.size

    # Añadir Sal (píxeles blancos)
    num_salt = int(num_pixels * amount * 0.5)
    coords_salt = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy_img[coords_salt[0], coords_salt[1]] = 255

    # Añadir Pimienta (píxeles negros)
    num_pepper = int(num_pixels * amount * 0.5)
    coords_pepper = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy_img[coords_pepper[0], coords_pepper[1]] = 0

    return noisy_img

# --- 1. Cargar la Imagen --- (robusto: usa ruta absoluta + fallback a Pillow)
print("Buscando 'imagen.webp' junto al script...")

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'imagen.webp')
print("Ruta comprobada:", image_path)

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
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
        print("Compruebe que 'imagen.webp' no esté corrupta o conviértala a PNG/JPG.")
        sys.exit(1)
else:
    print("Imagen cargada correctamente con OpenCV.")

# Convertir a escala de grises para simplificar el filtrado
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 2. Añadir Ruido ---
print("Añadiendo ruido de 'sal y pimienta'...")
# Añadimos un 5% de ruido
noisy_image = add_salt_and_pepper(img_gray, 0.05)

# --- 3. Aplicar el Filtro de Mediana ---
print("Aplicando filtro de mediana (Kernel 5x5)...")
# cv2.medianBlur(imagen_ruidosa, tamaño_del_kernel)
# El tamaño del kernel debe ser un número impar (ej: 3, 5, 7)
# Un kernel 5x5 mira una vecindad de 5x5 píxeles.
filtered_image = cv2.medianBlur(noisy_image, 5)

# --- 4. Mostrar Resultados ---
print("Mostrando resultados...")

# Usamos matplotlib para mostrar las 3 imágenes juntas
fig, axs = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle('Preprocesado: Filtro de Mediana para Ruido de Sal y Pimienta', fontsize=16)

# Imagen Original
axs[0].imshow(img_gray, cmap='gray')
axs[0].set_title('Original (Escala de Grises)')
axs[0].axis('off')

# Imagen con Ruido
axs[1].imshow(noisy_image, cmap='gray')
axs[1].set_title('Con Ruido (Sal y Pimienta)')
axs[1].axis('off')

# Imagen Filtrada
axs[2].imshow(filtered_image, cmap='gray')
axs[2].set_title('Filtrada (Mediana 5x5)')
axs[2].axis('off')

plt.tight_layout()
plt.show()

print("Proceso completado.")