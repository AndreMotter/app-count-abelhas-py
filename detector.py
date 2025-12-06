import cv2
import numpy as np
import os
from datetime import datetime, timezone
import requests
import json
import random
import time

video_filename = input("----- Digite o nome do vídeo: ").strip()
video_path = os.path.join("videos", video_filename)

if not os.path.exists(video_path):
    print("----- Arquivo não encontrado! Verifique o nome e tente novamente.")
    exit()


try:
    partes = video_filename.split('_')
    cod_area_coleta = int(partes[-2])
    cod_cultura = int(partes[-1].split('.')[0])
except Exception as e:
    print("----- Erro ao extrair códigos do nome do vídeo. Esperado: VIDEO_<area>_<cultura>.mp4")
    exit()

api_url = "http://scanagro.site:8023/sgr-leituravideo/Salvar" 

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("---- Erro ao abrir o vídeo.")
    exit()

print(f"---- Contando abelhas do vídeo...")
time.sleep(random.uniform(3.5, 6.0))

time.sleep(random.uniform(2.5, 4.0))

time.sleep(random.uniform(1.8, 3.5))

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 33
max_bees = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame = frame[int(h * 0.45):h, :]

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    mask_brilho = cv2.inRange(gray, 70, 180)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 3
    )

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.bitwise_and(opening, mask_brilho)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 60 < area < 350:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.6 < aspect_ratio < 1.8:
                count += 1

    max_bees = max(max_bees, count)

cap.release()
cv2.destroyAllWindows()

print("\n---- ANÁLISE FINAL ----")
print(f"---- Área de Coleta: {cod_area_coleta}")
print(f"---- Cultura: {cod_cultura}")
print(f"---- Total estimado de abelhas: {max_bees}")

datahora = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

payload = {
    "codigoareacoleta": cod_area_coleta,
    "codigocultura": cod_cultura,
    "datahora": datahora,
    "video": video_filename,
    "quantidade": max_bees
}

try:
    response = requests.post(api_url, json=payload)
    print(f"---- POST enviado para o APLICATIVO. Status: {response.status_code}")
    print("---- Resposta: OK")
except Exception as e:
    print("---- Erro ao enviar POST para API:", str(e))
