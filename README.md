# 🔒 Face Authentication & Recognition System

An advanced, real-time facial recognition and login authentication system that utilizes state-of-the-art Deep Learning models. The system features face detection via **YOLOv5** and **MTCNN**, and face feature extraction (embedding generation) using **FaceNet** (`InceptionResnetV1` pre-trained on `vggface2`).

---

## 🛠️ System Architecture

1. **Face Detection**:
   - **MTCNN**: Multi-task Cascaded Convolutional Networks for lightning-fast and highly accurate face crop extraction.
   - **YOLOv5-Face**: High-performance bounding-box localization for multi-face tracking in dense environments.
2. **Feature Extraction**:
   - **FaceNet (InceptionResnetV1)**: Translates the face crop into a high-dimensional (512-D) vector space representation (embedding).
3. **Face Verification**:
   - L2-distance comparison against registered user templates. Minimum Euclidean distance beneath threshold (e.g., `0.8`) registers a successful login.

---

## 📂 Project Structure

- 📁 **`dataset/`**: Registry directory where face pictures are stored inside folders named after the users (e.g., `dataset/john_doe/img1.jpg`).
- 📄 **`extract_embeddings.py`**: Reads face images from `dataset/`, runs MTCNN crops, extracts FaceNet vectors, and stores them in `face_embeddings.pkl`.
- 📄 **`face_login_web.py`**: A beautiful Flask web server implementing a high-performance webcam capture feed and API for face verification.
- 📄 **`face_login.py`**: Desktop camera script that runs face verification directly inside an OpenCV window.
- 📄 **`app.py`**: Web-based YOLOv5-integrated multi-face tracking and recognition portal.
- 📄 **`requirements.txt`**: Package requirements list.

---

## 🚀 Setup & Execution

### 1. Install Dependencies
Ensure you have Python 3.11+ installed. Run:
```bash
pip install -r requirements.txt
```

### 2. Extract Embeddings (Seeding Users)
Organize your face images in the `dataset/` folder, then extract embeddings by running:
```bash
python extract_embeddings.py
```
This saves database embeddings in `face_embeddings.pkl`.

### 3. Run Web Login Portal (Flask)
For a premium web application login interface with a live camera preview:
```bash
python face_login_web.py
```
- Open `http://127.0.0.1:5000` in your web browser.
- Click **"Capture Image"** then **"Recognize"** to sign in!

### 4. Run CLI Camera Login (OpenCV)
For direct system terminal login using OpenCV:
```bash
python face_login.py
```
- The camera window will pop up. 
- Look at the camera; it will verify your face, print a success message, and exit upon recognition!

---

## 📝 Important Developer Notes
- **YOLOv5 Repo Integration**: The `app.py` script requires local YOLOv5 files to be present in the directory (`yolov5/`). We patched hardcoded user directory links to use relative, portable environment pathways.
- **Model Weights**: Model files `yolov5s-face.pt` and `yolov5s.pt` provide face classification metrics. Ensure they are in the root of the project.
