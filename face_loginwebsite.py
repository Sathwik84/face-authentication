import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

# Load YOLOv5 model (pre-trained on COCO)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

def detect_with_yolo(frame):
    # YOLO expects RGB images
    results = yolo_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
    return detections

def detect_faces_with_mtcnn(frame):
    boxes, _ = mtcnn.detect(frame)
    return boxes

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. YOLO detection (find persons)
        detections = detect_with_yolo(frame)

        # Filter for 'person' class (class 0 in COCO)
        person_boxes = [det for det in detections if int(det[5]) == 0 and det[4] > 0.4]

        # 2. For each person, run MTCNN for face detection
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            face_boxes = detect_faces_with_mtcnn(person_crop)
            # Draw person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Draw face boxes within person
            if face_boxes is not None:
                for fbox in face_boxes:
                    fx1, fy1, fx2, fy2 = map(int, fbox)
                    cv2.rectangle(frame, (x1+fx1, y1+fy1), (x1+fx2, y1+fy2), (0, 255, 0), 2)

        # (Optional) Also run MTCNN on full frame for extra faces not inside persons
        all_face_boxes = detect_faces_with_mtcnn(frame)
        if all_face_boxes is not None:
            for fbox in all_face_boxes:
                x1, y1, x2, y2 = map(int, fbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imshow('Hybrid YOLO+MTCNN Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
