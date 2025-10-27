import numpy as np
import cv2
import os

print("Iniciando Flujo Óptico (Lucas-Kanade)...")
print("Mueve tu mano u un objeto frente a la cámara.")
print("Presiona 'q' para salir.")

# --- 1. Abrir la Webcam ---
cap = cv2.VideoCapture(0) # 0 es el índice de tu webcam por defecto

if not cap.isOpened():
    print("Error: No se pudo abrir la webcam.")
    exit()

# --- 2. Parámetros para la detección de esquinas (Shi-Tomasi) ---
feature_params = dict(
    maxCorners=100,      # Máximo 100 esquinas a rastrear
    qualityLevel=0.3,    # Umbral de calidad (de 0 a 1)
    minDistance=7,       # Distancia mínima entre esquinas
    blockSize=7
)

# --- 3. Parámetros para el Flujo Óptico (Lucas-Kanade) ---
lk_params = dict(
    winSize=(15, 15),    # Tamaño de la ventana de búsqueda
    maxLevel=2,          # Niveles de la pirámide de imágenes
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Crear algunos colores aleatorios para dibujar
color = np.random.randint(0, 255, (100, 3))

# --- 4. Tomar el primer fotograma y encontrar esquinas ---
ret, old_frame = cap.read()
if not ret:
    print("Error: No se pudo leer el primer fotograma.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Crear una máscara (imagen en negro) para dibujar las estelas
mask = np.zeros_like(old_frame)

# --- 5. Bucle Principal (Procesar Video) ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del stream.")
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- Calcular el Flujo Óptico ---
    # p1 = nuevas posiciones de las esquinas
    # st = status (1 si el punto se rastreó, 0 si se perdió)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Seleccionar solo los "buenos puntos" (los que se encontraron)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
    # --- Dibujar las Estelas (Tracks) ---
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        
        # Dibujar la línea de estela en la máscara
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        # Dibujar el punto actual en el fotograma
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    
    # Combinar el fotograma con la máscara de estelas
    img_with_flow = cv2.add(frame, mask)
    
    # Mostrar el resultado
    cv2.imshow('Flujo Óptico (Movimiento)', img_with_flow)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
        
    # Actualizar el fotograma anterior y los puntos anteriores
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# --- 6. Limpieza ---
print("Cerrando...")
cap.release()
cv2.destroyAllWindows()