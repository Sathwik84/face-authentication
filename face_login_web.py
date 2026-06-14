
import io
import base64
import torch
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template_string, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1

# Load embeddings and labels
with open('face_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
known_embeddings = np.array(data['embeddings'])
known_labels = data['labels']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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
        body {
            background: #f0f2f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .container {
            max-width: 480px;
            margin-top: 60px;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        video, img {
            border-radius: 12px;
            width: 100%;
            height: auto;
            max-height: 360px;
            object-fit: cover;
            border: 2px solid #0d6efd;
        }
        #message {
            min-height: 40px;
            margin-top: 20px;
        }
        button {
            min-width: 140px;
        }
        .btn-group {
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
        }
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

    // Access webcam
    navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } })
        .then(stream => { video.srcObject = stream; })
        .catch(err => { messageDiv.innerHTML = `<div class="alert alert-danger">Webcam error: ${err.message}</div>`; });

    // Capture image
    captureBtn.addEventListener('click', () => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        capturedDataURL = canvas.toDataURL('image/jpeg');
        resultImg.src = capturedDataURL;
        recognizeBtn.disabled = false;
        messageDiv.innerHTML = '';
    });

    // Recognize face
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
    data = request.get_json()
    img_data = data['image'].split(',')[1]
    img_bytes = io.BytesIO(base64.b64decode(img_data))
    img = Image.open(img_bytes).convert('RGB')

    # Detect face
    face = mtcnn(img)
    if face is None:
        return jsonify({'success': False, 'message': 'No face detected. Please try again.', 'processed_image': ''})

    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device)).squeeze().cpu().numpy()
    name = recognize_face(embedding, known_embeddings, known_labels)

    # Draw bounding box and label on the original image for preview
    draw = ImageDraw.Draw(img)
    # Get bounding box from MTCNN (approximate from face tensor)
    box = mtcnn.detect(img)[0]
    if box is not None:
        x1, y1, x2, y2 = [int(b) for b in box[0]]
        draw.rectangle([x1, y1, x2, y2], outline='green', width=4)
        font = ImageFont.load_default()
        draw.text((x1, y1 - 15), name, fill='green', font=font)

    # Convert image to base64 to send back
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    if name == "Unknown":
        return jsonify({'success': False, 'message': 'Face not recognized.', 'processed_image': img_str})
    else:
        return jsonify({'success': True, 'message': f'Login successful! Welcome, {name}.', 'processed_image': img_str})

if __name__ == '__main__':
    app.run(debug=True)
