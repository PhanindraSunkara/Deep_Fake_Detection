# 🎭 Deepfake Detection System

The **Deepfake Detection System** is an AI-powered web application that detects manipulated (fake) videos using **Convolutional Neural Networks (CNN)** and **MTCNN** for face detection. The system extracts facial frames from videos, classifies them as **REAL** or **FAKE**, and displays the result via a **Flask-based web interface**.

---

## 🚀 Features

- Upload any `.mp4` video to check its authenticity  
- Detects facial regions using **MTCNN** and analyzes with a trained **CNN**  
- Predicts if the video is **REAL** or **FAKE** based on average frame classification  
- Displays results via a clean **Flask UI**  
- Tracks number of videos predicted as real or fake  
- Fast inference using **frame skipping** technique

---

## 🧰 Tech Stack

- **Python** for backend and model logic  
- **TensorFlow / Keras** for building and loading CNN model  
- **OpenCV** for frame extraction  
- **MTCNN** for face detection  
- **Flask** for web application  
- **HTML / CSS / Bootstrap** for frontend UI  
- **NumPy** for array processing  

---

## 📊 Dataset

The model was trained using a combination of popular deepfake datasets:

- **FaceForensics++**  
- **Celeb-DF (v2)**  
- **DFDC (Deepfake Detection Challenge)**

> These datasets include thousands of real and deepfake videos used to teach the model how to identify facial manipulations.

---

## ⚙️ Installation, Usage, & Output

```bash
# Step 1: Clone the repository
git clone https://github.com/YourUsername/Deepfake-Detection-System.git
cd Deepfake-Detection-System

# Step 2: Install dependencies
pip install -r requirements.txt
# Or manually:
# pip install flask tensorflow opencv-python mtcnn numpy

# Step 3: Add the trained model
# Place your trained_model.h5 in the root directory

# Step 4: Run the Flask web application
python app.py

# Visit in your browser:
# http://127.0.0.1:5000

# How It Works:
# 1. The user uploads a video through the Flask interface
# 2. Frames are extracted using OpenCV
# 3. MTCNN detects and crops face regions from the frames
# 4. Each face is resized, normalized, and passed to a CNN model
# 5. CNN gives a probability score for each face
# 6. Average score is calculated to classify the video as REAL or FAKE
#    (score > 0.5 = FAKE, otherwise REAL)

# Example Output:
# 🎥 Uploaded Video: celebrity_clip.mp4
# 🧑 Faces Detected: 24
# 📊 Average Confidence: 0.84
# 🎯 Final Prediction: FAKE
# 📈 Total FAKE videos so far: 5
