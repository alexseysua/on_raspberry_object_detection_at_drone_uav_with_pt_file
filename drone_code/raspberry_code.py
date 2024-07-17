from ultralytics import YOLO
import cv2
import numpy as np
import serial

# XBee için seri portu yapılandırma
ser = serial.Serial('/dev/ttyUSB0', 9600)  # Raspberry Pi'deki XBee modülünün bağlı olduğu seri port

modello = YOLO("last.pt")

def yaz(file, result):
    coords = result[0].boxes.xyxy
    if coords.size(0) == 0:
        return
    labels = result[0].boxes.cls
    lbl_np = np.array(labels.cpu())
    a_np = np.array(coords.cpu())
    file.write("[\n")
    for lbl, arr in zip(lbl_np, a_np):
        arr_str = ' '.join(map(str, arr))  # Koordinatları string'e çevir
        file.write(f"{int(lbl)} {arr_str}\n")  # Etiketi ve koordinatları yaz
    file.write("]\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

with open("data.txt", "w") as file:
    counter = 0
    while True:
        ret, frame = cap.read()
        result = modello(frame, imgsz=(256))
        frame = result[0].plot()
        cv2.imshow("det", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        counter += 1
        if counter == 10:
            counter = 0
            yaz(file, result)

cap.release()
cv2.destroyAllWindows()

# Dosya yazma işlemi tamamlandıktan sonra dosyayı oku ve XBee üzerinden gönder
with open("data.txt", "r") as file:
    data = file.read()
    ser.write(data.encode('utf-8'))

ser.close()
