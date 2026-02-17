import io
import base64
import torch
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template_string, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

# Load known embeddings and labels
with open('face_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
known_embeddings = np.array(data['embeddings'])
known_labels = data['labels']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN and FaceNet models
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Set paths to YOLOv5 repo and weights
YOLOV5_REPO = r"C:\Users\manik\Desktop\project dupe\yolov5"  # Update this to your yolov5 repo path
WEIGHTS_PATH = os.path.join(os.getcwd(), 'yolov5s-face.pt')  # Assuming weights in current folder

# Load YOLOv5 face detection model via torch.hub from local repo
yolo_model = torch.hub.load(YOLOV5_REPO, 'custom', path=WEIGHTS_PATH, source='local')

def recognize_face(embedding, known_embeddings, known_labels, threshold=0.8):
    distances = np.linalg.norm(known_embeddings - embedding, axis=1)
    min_dist_idx = np.argmin(distances)
    if distances[min_dist_idx] < threshold:
        return known_labels[min_dist_idx]
    else:
        return "Unknown"

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Face Recognition Login</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
    <style>
        body { background: #f0f2f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .container { max-width: 480px; margin-top: 60px; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
        video, img { border-radius: 12px; width: 100%; height: auto; max-height: 360px; object-fit: cover; border: 2px solid #0d6efd; }
        #message { min-height: 40px; margin-top: 20px; }
        button { min-width: 140px; }
        .btn-group { display: flex; justify-content: space-between; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="container shadow-sm">
        <h2 class="text-center mb-4 text-primary">Face Recognition Login</h2>
        <video id="video" autoplay playsinline></video>
        <div class="btn-group mt-3">
            <button id="capture-btn" class="btn btn-outline-primary">Capture Image</button>
            <button id="recognize-btn" class="btn btn-primary" disabled>Recognize</button>
        </div>
        <canvas id="canvas" width="480" height="360" style="display:none;"></canvas>
        <img id="result-img" alt="Captured Image Preview" class="mt-3 rounded shadow-sm" />
        <div id="message" class="text-center"></div>
    </div>
<script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const recognizeBtn = document.getElementById('recognize-btn');
    const resultImg = document.getElementById('result-img');
    const messageDiv = document.getElementById('message');
    const ctx = canvas.getContext('2d');
    let capturedDataURL = null;

    navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } })
        .then(stream => { video.srcObject = stream; })
        .catch(err => { messageDiv.innerHTML = `<div class="alert alert-danger">Webcam error: ${err.message}</div>`; });

    captureBtn.addEventListener('click', () => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        capturedDataURL = canvas.toDataURL('image/jpeg');
        resultImg.src = capturedDataURL;
        recognizeBtn.disabled = false;
        messageDiv.innerHTML = '';
    });

    recognizeBtn.addEventListener('click', () => {
        if (!capturedDataURL) return;
        messageDiv.innerHTML = '<div class="spinner-border text-primary" role="status"></div>';
        fetch('/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: capturedDataURL })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                messageDiv.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
            } else {
                messageDiv.innerHTML = `<div class="alert alert-danger">${data.message}</div>`;
            }
            resultImg.src = 'data:image/jpeg;base64,' + data.processed_image;
        })
        .catch(() => {
            messageDiv.innerHTML = `<div class="alert alert-danger">Error during recognition.</div>`;
        });
    });
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        img_data = data['image'].split(',')[1]
        img_bytes = io.BytesIO(base64.b64decode(img_data))
        pil_img = Image.open(img_bytes).convert('RGB')
        img_np = np.array(pil_img)

        # Detect faces with YOLOv5
        results = yolo_model(img_np)
        boxes = results.xyxy[0].cpu().numpy() if results.xyxy is not None else []

        recognized_faces = []
        draw = ImageDraw.Draw(pil_img)

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            margin = 15  # increased margin for better MTCNN detection
            x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(img_np.shape[1], x2 + margin)
            y2m = min(img_np.shape[0], y2 + margin)

            face_roi = pil_img.crop((x1m, y1m, x2m, y2m))

            face_tensor = mtcnn(face_roi)
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
            bbox = draw.textbbox((0, 0), name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 10, y1], fill='green')
            draw.text((x1 + 5, y1 - text_height - 2), name, fill='white', font=font)

            recognized_faces.append(name)

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        if recognized_faces:
            return jsonify({'success': True, 'message': f'Recognized: {", ".join(set(recognized_faces))}', 'processed_image': img_str})
        else:
            return jsonify({'success': False, 'message': 'No faces recognized.', 'processed_image': img_str})

    except Exception as e:
        import traceback
        print(f"Recognition error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Error during recognition.', 'processed_image': ''})

if __name__ == '__main__':
    app.run(debug=True)
