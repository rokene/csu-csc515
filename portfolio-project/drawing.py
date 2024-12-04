#!/usr/bin/env python3

import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture the image or 'q' to quit.")

while True:
    # Read frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Display the live video feed
    cv2.imshow('Press "c" to Capture', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Capture the current frame
        captured_image = frame.copy()
        print("Image captured.")
        break
    elif key == ord('q'):
        print("Quitting without capturing.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Release the webcam and close the live feed window
cap.release()
cv2.destroyAllWindows()

# Convert the captured image to grayscale
gray = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

if len(faces) == 0:
    print("No face detected.")
    exit()

# Assuming the largest detected face is the target
face = max(faces, key=lambda rect: rect[2] * rect[3])
(x, y, w, h) = face

# Detect eyes within the face region
roi_gray = gray[y:y + h, x:x + w]
roi_color = captured_image[y:y + h, x:x + w]
eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

if len(eyes) < 2:
    print("Less than two eyes detected.")

# Draw a green circle around the face
center_x = x + w // 2
center_y = y + h // 2
radius = int(0.6 * (w + h) / 4)  # Adjust the radius as needed
cv2.circle(captured_image, (center_x, center_y), radius, (0, 255, 0), 2)

# Draw red bounding boxes around the eyes
for (ex, ey, ew, eh) in eyes[:2]:  # Consider only the first two detected eyes
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
cv2.imwrite('annotated_selfie.jpg', captured_image)
print("Annotated image saved as 'annotated_selfie.jpg'.")

# Wait until a key is pressed and then close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
