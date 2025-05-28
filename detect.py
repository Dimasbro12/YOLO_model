import cv2
from ultralytics import YOLO

model_path = "best.pt"  
image_path = "test1.jpeg"                   
output_path = "hasil_deteksi1.jpg"                

model = YOLO(model_path)

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

results = model(image)[0]

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf)
    cls = int(box.cls[0])
    label = model.names[cls]

    if conf > 0.5:  
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Deteksi Topi - YOLOv8", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(output_path, image)
print(f"✅ Hasil deteksi disimpan ke: {output_path}")