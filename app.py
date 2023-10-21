from flask import Flask, render_template, Response
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvzone

# Initializing Flask application
app = Flask(__name__)

# Setting up the color shades for display
shade1 = (247, 247, 67)
shade2 = (109, 247, 67)
shade3 = (70, 255, 225)
shade4 = (70, 150, 255)
orange = (0, 69, 255)
blue = (255, 0, 0)

# Load pre-trained face detection model
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the pre-trained face mask detection model
maskNet = load_model("mask_detector.model")

# Initialize the video stream
print("camera started...")
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width
cap.set(4, 480)  # Set the height

def detect_and_predict_mask(frame, faceNet, maskNet):
    """
    Detects faces in a frame and predicts if each face is wearing a mask.

    Parameters:
    - frame: A frame from the video capture
    - faceNet: Pre-trained face detection model
    - maskNet: Pre-trained mask detection model

    Returns:
    - locs: Locations of detected faces
    - preds: Predictions if face is wearing a mask
    """

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # Iterate over the detections and filter out weak detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            # Extract the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def generate_frames():
    """
    Generator function to produce frames for display in Flask.
    Processes each frame to detect faces and predict if they're wearing masks.
    """
    global cap, faceNet, maskNet
    while True:
        success, frame = cap.read()

        frame_width = 800
        frame_height = int(frame_width * (frame.shape[0] / frame.shape[1]))
        frame = cv2.resize(frame, (frame_width, frame_height))

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # Drawing on the frame based on predictions
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = blue if label == "Mask" else orange
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cvzone.putTextRect(frame, label, (startX, startY - 20), scale=1.5, thickness=2, colorT=color,
                               colorR=shade2, font=cv2.FONT_HERSHEY_PLAIN, offset=12, border=1, colorB=shade3)
            cvzone.cornerRect(frame, (startX, startY, endX - startX, endY - startY), l=30, t=5, rt=1,
                              colorR=(255, 0, 255), colorC=(0, 255, 0))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Route to render the HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video feed route to stream live video with face mask detection."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
