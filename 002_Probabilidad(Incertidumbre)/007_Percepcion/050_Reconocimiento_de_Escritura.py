import easyocr
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# --- 1. Cargar la Imagen de Prueba ---
# (Usaremos la 'ocr_test.png' que ya creaste)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'ocr_test.png')
    
    img_pil = Image.open(image_path)
    
except FileNotFoundError:
    print(f"Error: No se pudo cargar la imagen 'ocr_test.png'.")
    print(f"Asegúrate de haberla creado en la carpeta: {script_dir}")
    exit()

print(f"Cargando imagen: {image_path}")

# --- 2. Inicializar el Lector (Reader) ---
#
# ¡Esto es lo más importante!
# Le pedimos que cargue los modelos para español ('es').
# La PRIMERA VEZ que ejecutes esto, TARDARÁ en descargar los modelos.
print("Cargando modelo 'easyocr' para español (la primera vez puede tardar)...")
reader = easyocr.Reader(['es']) 
print("Modelo cargado.")

# --- 3. Extraer el Texto (OCR) ---
print("Procesando imagen con EasyOCR...")

# .readtext() devuelve una lista de resultados
# Cada resultado es [bounding_box, texto, confianza]
results = reader.readtext(image_path)

# --- 4. Mostrar Resultados ---
print("\n--- Texto Extraído ---")
if not results:
    print("(EasyOCR no pudo reconocer ningún texto.)")
else:
    draw = ImageDraw.Draw(img_pil)
    
    for (bbox, text, prob) in results:
        # Imprimir en la terminal
        print(f"  Texto: '{text}' (Confianza: {prob:.4f})")
        
        # Dibujar los cuadros (bounding boxes) en la imagen
        # bbox es una lista de 4 puntos [top-left, top-right, bottom-right, bottom-left]
        top_left = tuple(bbox[0])
        bottom_right = tuple(bbox[2])
        draw.rectangle([top_left, bottom_right], outline='red', width=2)
        draw.text(top_left, text, fill='red')

# --- 5. Mostrar la imagen con los resultados ---
print("\nMostrando imagen con texto detectado...")
plt.figure(figsize=(10, 8))
plt.imshow(img_pil)
plt.title('Reconocimiento con EasyOCR')
plt.axis('off')
plt.show()

print("Proceso completado.")