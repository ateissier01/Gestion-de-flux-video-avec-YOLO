import cv2
from ultralytics import YOLO

url="https://zssd-elephant-2025.hls.camzonecdn.com/CamzoneStreams/zssd-elephant-2025/Playlist.m3u8"

cap = cv2.VideoCapture(url)
model = YOLO("yolo11n-seg.pt")

if not cap.isOpened():
    print("Impossible d’ouvrir le flux")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame non reçue")
        break

    # 3. Inférence YOLO
    results = model(frame, conf=0.4, device="cpu")  # device="cuda" si GPU

    # 4. Dessiner les résultats
    annotated_frame = results[0].plot()

    # 5. Afficher
    cv2.imshow("YOLO - Webcam Zoo", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
