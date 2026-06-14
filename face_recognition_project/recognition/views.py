import io
import base64
import torch
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
from django.shortcuts import render
from django.http import JsonResponse
import os

# Load known face data
with open('face_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
known_embeddings = np.array(data['embeddings'])
known_labels = data['labels']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
from facenet_pytorch import MTCNN, InceptionResnetV1
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load YOLOv5 model
YOLOV5_REPO = os.path.join(os.getcwd(), 'yolov5')
WEIGHTS_PATH = os.path.join(os.getcwd(), 'yolov5s-face.pt')
yolo_model = torch.hub.load(YOLOV5_REPO, 'custom', path=WEIGHTS_PATH, source='local')

def recognize_face(embedding, known_embeddings, known_labels, threshold=0.8):
    distances = np.linalg.norm(known_embeddings - embedding, axis=1)
    min_dist_idx = np.argmin(distances)
    if distances[min_dist_idx] < threshold:
        return known_labels[min_dist_idx]
    return "Unknown"

def index(request):
    return render(request, 'recognition/index.html')

def recognize(request):
    try:
        data = request.json() if request.content_type == 'application/json' else request.POST
        image_data = data.get('image').split(',')[1]
        img_bytes = io.BytesIO(base64.b64decode(image_data))
        pil_img = Image.open(img_bytes).convert('RGB')
        img_np = np.array(pil_img)

        results = yolo_model(img_np)
        boxes = results.xyxy[0].cpu().numpy() if results.xyxy is not None else []

        draw = ImageDraw.Draw(pil_img)
        recognized_faces = []

        for box in boxes:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            face = pil_img.crop((x1, y1, x2, y2))
            face_tensor = mtcnn(face)
            if face_tensor is None:
                continue
            with torch.no_grad():
                embedding = resnet(face_tensor.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            name = recognize_face(embedding, known_embeddings, known_labels)

            draw.rectangle([x1, y1, x2, y2], outline='green', width=4)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            draw.text((x1, y1 - 20), name, fill='white', font=font)

            recognized_faces.append(name)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return JsonResponse({
            'success': True,
            'message': ', '.join(recognized_faces) if recognized_faces else 'No faces recognized.',
            'processed_image': img_str
        })
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e), 'processed_image': ''})
