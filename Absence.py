import cv2
import numpy as np

# Load YOLOv4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names (label classes)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Inisialisasi Video Capture (kamera laptop)
cap = cv2.VideoCapture(0)

# Deteksi hanya kelas "person"
TARGET_CLASS = "person"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Preprocessing untuk YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Parsing hasil deteksi
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == TARGET_CLASS:  # Hanya deteksi 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Max Suppression untuk menghindari box duplikat
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Menggambar box hijau
    count = 0  # Jumlah mahasiswa terdeteksi
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

    # Tampilkan jumlah mahasiswa
    cv2.putText(frame, f"Jumlah Mahasiswa: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    people_label = "Person Detected" if boxes else "No Person"
    cv2.putText(frame, people_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Deteksi Mahasiswa", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
