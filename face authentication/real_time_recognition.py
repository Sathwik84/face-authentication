import cv2
import torch
import numpy as np
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load saved embeddings and labels
with open('face_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
known_embeddings = np.array(data['embeddings'])
known_labels = data['labels']

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0)

# Initialize FaceNet model for embedding extraction
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def recognize_face(embedding, known_embeddings, known_labels, threshold=0.8):
    distances = np.linalg.norm(known_embeddings - embedding, axis=1)
    min_dist_idx = np.argmin(distances)
    if distances[min_dist_idx] < threshold:
        return known_labels[min_dist_idx]
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape

    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]

            # Clamp coordinates to image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            # Validate box
            if x2 <= x1 or y2 <= y1:
                continue

            face = img_rgb[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Resize face to 160x160 as required by FaceNet
            face = cv2.resize(face, (160, 160))

            # Convert to tensor and normalize
            face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255.0
            face_tensor = face_tensor.unsqueeze(0)

            with torch.no_grad():
                embedding = resnet(face_tensor).squeeze().numpy()

            name = recognize_face(embedding, known_embeddings, known_labels)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
