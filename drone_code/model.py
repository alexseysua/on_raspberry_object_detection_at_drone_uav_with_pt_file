from ultralytics import YOLO
import cv2 
import numpy as np 

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


with open("data.txt", "w") as file:
    counter = 0
    while True:        
        ret, frame = cap.read()
        result = modello(frame, imgsz=(480), conf = .45)
        cv2.imshow("det", result[0].plot())
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        counter+=1
        if counter == 40:
            counter = 0             
            yaz(file, result)

file.close()
cap.release()
cv2.destroyAllWindows()