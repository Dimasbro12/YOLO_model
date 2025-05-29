import cv2
import os
import numpy as np
from ultralytics import YOLO

model_path = "best.pt"
image_path = "IKI.jpg"
output_dir = "output_yolo"
conf_threshold = 0.5
grid_size = 7

os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")
height, width, _ = image.shape
results = model(image)[0]

step1 = image.copy()
cv2.imwrite(f"{output_dir}/step1_input.jpg", step1)
cv2.imshow("Step 1: Input Image", step1)
cv2.waitKey(0)

def draw_grid(img, S=7):
    h, w, _ = img.shape
    for i in range(1, S):
        cv2.line(img, (0, i * h // S), (w, i * h // S), (255, 255, 255), 1)
        cv2.line(img, (i * w // S, 0), (i * w // S, h), (255, 255, 255), 1)
    return img

step2 = draw_grid(image.copy(), S=grid_size)
cv2.imwrite(f"{output_dir}/step2_grid.jpg", step2)
cv2.imshow("Step 2: Grid Overlay", step2)
cv2.waitKey(0)

def class_probability_map(image, results, grid_size=7):
    h, w, _ = image.shape
    step_y, step_x = h // grid_size, w // grid_size
    overlay = image.copy()

    # Warna untuk setiap kelas
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (0, 128, 128), (128, 0, 128),
        (255, 165, 0), (0, 100, 0), (75, 0, 130),
        (255, 20, 147), (255, 192, 203), (192, 192, 192),
        (139, 69, 19), (0, 255, 127)
    ]

    prob_map = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    for box in results.boxes:
        conf = float(box.conf)
        if conf > conf_threshold:
            x_center, y_center = map(int, box.xywh[0][:2])
            cls = int(box.cls[0])
            grid_x = min(grid_size - 1, x_center * grid_size // w)
            grid_y = min(grid_size - 1, y_center * grid_size // h)
            color = colors[cls % len(colors)]
            prob_map[grid_y, grid_x] = color

    alpha = 0.4
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * step_x
            y1 = row * step_y
            x2 = x1 + step_x
            y2 = y1 + step_y
            color = tuple(int(c) for c in prob_map[row, col])
            if color != (0, 0, 0):  # hindari sel kosong
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    blended = cv2.addWeighted(overlay, alpha, image.copy(), 1 - alpha, 0)
    blended = draw_grid(blended, grid_size)
    return blended

step3 = class_probability_map(image.copy(), results, grid_size)
cv2.imwrite(f"{output_dir}/step3_class_probability_map.jpg", step3)
cv2.imshow("Step 3: Class Probability Map", step3)
cv2.waitKey(0)

step4 = image.copy()
for box in results.boxes:
    conf = float(box.conf)
    if conf > conf_threshold:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(step3, (x1, y1), (x2, y2), (0, 255, 255), 2)
cv2.imwrite(f"{output_dir}/step4_bbox_only.jpg", step3)
cv2.imshow("Step 4: Bounding Boxes Only", step3)
cv2.waitKey(0)

step5 = image.copy()
for box in results.boxes:
    conf = float(box.conf)
    if conf > conf_threshold:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        cv2.rectangle(step5, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(step5, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite(f"{output_dir}/step5_bbox_labeled.jpg", step5)
cv2.imshow("Step 4: Labeled Detection", step5)
cv2.waitKey(0)

cv2.destroyAllWindows()
print("✅ Semua tahap visualisasi YOLO selesai. Cek folder:", output_dir)
