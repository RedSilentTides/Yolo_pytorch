import cv2
import cvzone
import math
from ultralytics import YOLO

# --- MODIFICACIÓN AQUÍ ---
# Cambiamos 'fall.mp4' por 0 para usar la cámara web principal
cap = cv2.VideoCapture(0) 

model = YOLO('yolov8s.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()


while True:
    ret, frame = cap.read()
    
    # --- AÑADIDO (BUENA PRÁCTICA) ---
    # Verificar si el frame se capturó correctamente
    if not ret:
        print("Error: No se pudo capturar el frame de la cámara.")
        break

    # Mantienes tu reescalado
    frame = cv2.resize(frame, (980,740))

    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)
                
                # --- LÓGICA MEJORADA ---
                # Tiene más sentido revisar la caída solo si detectaste una persona
                if threshold < 0:
                    # Ajusté la posición del texto para que aparezca bajo la caja
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 + height + 20], thickness=2, scale=2, colorR=(255,0,0))
            
            # (El 'else: pass' no es necesario y se puede omitir)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# --- AÑADIDO (BUENA PRÁCTICA) ---
# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()