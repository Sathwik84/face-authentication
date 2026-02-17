import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import pickle

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=0)

# Initialize InceptionResnetV1 for face embedding extraction
resnet = InceptionResnetV1(pretrained='vggface2').eval()

dataset_dir = r'C:\Users\kapar\Downloads\project-dupe\dataset'
  # path to your dataset folder
embeddings = []
labels = []

for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        img = Image.open(image_path).convert('RGB')
        
        # Detect face and crop
        face = mtcnn(img)
        if face is not None:
            # Get embedding
            with torch.no_grad():
                embedding = resnet(face.unsqueeze(0))
            embeddings.append(embedding.squeeze().numpy())
            labels.append(person_name)
        else:
            print(f"Face not detected in {image_path}")

# Save embeddings and labels
data = {'embeddings': embeddings, 'labels': labels}
with open('face_embeddings.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Face embeddings extracted and saved successfully!")
