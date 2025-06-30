from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

detector = MTCNN()
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.load_weights('trained_model.h5')

fake_count = 0
real_count = 0

def extract_faces_from_video(video_path, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = detector.detect_faces(frame_rgb)

        for face in detected_faces:
            x, y, width, height = face['box']
            face_img = frame_rgb[y:y+height, x:x+width]
            face_img = cv2.resize(face_img, target_size)
            face_array = img_to_array(face_img) / 255.0
            faces.append(face_array)
    
    cap.release()
    return np.array(faces)

def predict_video(video_path):
    faces = extract_faces_from_video(video_path)
    if len(faces) == 0:
        return "No faces detected in the video."

    faces = np.reshape(faces, (faces.shape[0], faces.shape[1], faces.shape[2], faces.shape[3]))

    predictions = model.predict(faces)
    avg_prediction = np.mean(predictions)

    if avg_prediction > 0.5:
        return "FAKE"
    else:
        return "REAL"

@app.route('/', methods=['GET', 'POST'])
def index():
    global fake_count, real_count
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            result = predict_video(file_path)
            if result == "FAKE":
                fake_count += 1
            elif result == "REAL":
                real_count += 1
            return render_template('result.html', result=result, fake_count=fake_count, real_count=real_count)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)