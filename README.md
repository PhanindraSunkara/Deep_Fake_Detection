Deepfake Detection System 🎭

This project is a deep learning-based web application that detects deepfake videos. It uses MTCNN for face detection and a Convolutional Neural Network (CNN) to classify videos as *REAL* or *FAKE*. The system is deployed using a Flask web interface where users can upload videos and view results instantly.

## Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- MTCNN  
- Flask  
- HTML/CSS  

## How It Works
1. User uploads a video through the web interface.
2. Video frames are extracted using OpenCV.
3. Faces are detected using MTCNN.
4. Each face is resized, normalized, and passed to a CNN.
5. The CNN outputs a prediction: *REAL* or *FAKE*.
6. The result is shown to the user via the Flask app.

## How to Run
```bash
git clone https://github.com/your-username/deepfake-detection-system.git
cd deepfake-detection-system
pip install -r requirements.txt
python app.py

Then go to http://127.0.0.1:5000 and upload your video.

Output

The app displays whether the uploaded video is FAKE or REAL based on average frame prediction.
