#!/usr/bin/env python3

import cv2
import numpy as np

# Paths to the model files
modelFile = "data/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "data/deploy.prototxt"

# Check if model files exist
import os
if not os.path.exists(modelFile) or not os.path.exists(configFile):
    print("Model files not found. Please ensure 'res10_300x300_ssd_iter_140000.caffemodel' and 'deploy.prototxt' are in the current directory.")
    exit()

# Load the DNN model
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture the image or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    cv2.imshow('Press "c" to Capture', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        captured_image = frame.copy()
        print("Image captured.")
        break
    elif key == ord('q'):
        print("Quitting without capturing.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Prepare the image for DNN
(h, w) = captured_image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(captured_image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))

# Perform face detection
net.setInput(blob)
detections = net.forward()

# Initialize list of faces
faces = []

# Loop over the detections
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    # Filter out weak detections
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        faces.append((startX, startY, endX - startX, endY - startY))
        # Draw the bounding box
        cv2.rectangle(captured_image, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)

# Check if any faces were detected
if len(faces) == 0:
    print("No face detected.")
    exit()

# Assuming the first detected face is the target
(x, y, w, h) = faces[0]

# Draw a green circle around the face
center_x = x + w // 2
center_y = y + h // 2
radius = int(0.6 * (w + h) / 4)  # Adjust the radius as needed
cv2.circle(captured_image, (center_x, center_y), radius, (0, 255, 0), 2)

# Detect eyes within the face region using Haar Cascades
gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
roi_gray = gray[y:y + h, x:x + w]
eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

if len(eyes) < 2:
    print("Less than two eyes detected.")
else:
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_x = x + ex
        eye_y = y + ey
        cv2.rectangle(captured_image, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (0, 0, 255), 2)

# Add the text tag "this is me" below the face
font = cv2.FONT_HERSHEY_SIMPLEX
text = "this is me"
text_size, _ = cv2.getTextSize(text, font, 1, 2)
text_x = x + w // 2 - text_size[0] // 2
text_y = y + h + 30  # Adjust the y-coordinate as needed
cv2.putText(captured_image, text, (text_x, text_y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Display the annotated image
cv2.imshow('Annotated Image', captured_image)

# Save the annotated image to disk
cv2.imwrite('annotated_selfie_dnn.jpg', captured_image)
print("Annotated image saved as 'annotated_selfie_dnn.jpg'.")

# Wait until a key is pressed and then close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
