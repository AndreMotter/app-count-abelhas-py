import cv2
import numpy as np

print("---- Buscando vídeo...")

video_path = "videos/video_01.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

    print("---- Vídeo carregado, iniciando YOLO...")

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33

max_bees = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === RECORTA A PARTE ONDE AS ABELHAS REALMENTE ESTÃO ===
    # (Ignora madeira superior e plantas)
    h, w = frame.shape[:2]
    frame = frame[int(h * 0.45):h, :]  # usa só metade inferior do vídeo

    # Reduz ruído
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Cinza + Aumento de contraste (CLAHE)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Máscara para brilho médio (abelhas são marrom-amareladas)
    mask_brilho = cv2.inRange(gray, 70, 180)  # ajuste fino aqui se precisar

    # Threshold para separar objetos do fundo
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 3
    )

    # Remove ruídos pequenos
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # APLICA FILTRO DE BRILHO
    opening = cv2.bitwise_and(opening, mask_brilho)

    # Contornos
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 60 < area < 350:  # área ajustada

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Mantém apenas contornos com formato aproximado de abelha
            if 0.6 < aspect_ratio < 1.8:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                count += 1

    max_bees = max(max_bees, count)

    cv2.putText(frame, f"Abelhas detectadas (estimado): {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("Contador de Abelhas", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nTotal aproximado máximo de abelhas detectadas em um frame: {max_bees}")
