import requests
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import os

# --- 1. Cargar la Imagen ---
# Usamos 'test.jpg' (la imagen JPG que SÍ funcionaba)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'test.jpg')
    img = Image.open(image_path)
    if img is None:
        raise FileNotFoundError
except FileNotFoundError:
    print(f"Error: No se pudo cargar la imagen 'test.jpg' desde {image_path}")
    exit()

print("Imagen cargada exitosamente.")

# --- 2. Cargar el Modelo (¡Automáticamente!) ---
# Esto descargará el modelo la primera vez que lo ejecutes.
print("Cargando modelo DETR de Hugging Face (puede tardar)...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
print("Modelo cargado.")

# --- 3. Preparar y Procesar la Imagen ---
print("Detectando objetos...")
inputs = processor(images=img, return_tensors="pt")
outputs = model(**inputs)

# --- 4. Post-Procesamiento ---
# Convertir resultados a formato COCO (similar a YOLO)
# Umbral de confianza: solo aceptar detecciones > 90%
target_sizes = torch.tensor([img.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# --- 5. Dibujar los Resultados ---
draw = ImageDraw.Draw(img)

print(f"Objetos encontrados: {len(results['scores'])}")

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    # Obtener el nombre de la etiqueta (ej: 'cat')
    label_name = model.config.id2label[label.item()]
    
    print(f"  Encontrado: {label_name} (Confianza: {score:.2f})")
    
    # Dibujar el cuadro
    draw.rectangle(box.tolist(), outline="red", width=3)
    
    # Dibujar la etiqueta
    draw.text((box[0], box[1] - 10), f"{label_name} {score:.2f}", fill="red")

# --- 6. Mostrar la Imagen Final ---
print("Mostrando resultado...")
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.title('Reconocimiento de Objetos con DETR (Hugging Face)')
plt.axis('off')
plt.show()

print("Proceso completado.")