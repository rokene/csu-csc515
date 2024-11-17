#!/usr/bin/env python3

import cv2

# Create a named window with normal settings
cv2.namedWindow('Brain Image', cv2.WINDOW_NORMAL)

# Load and display the image
image = cv2.imread('brain.jpg')
cv2.imshow('Brain Image', image)

print('Press "q" to quit.')

# Keep the window open until 'q' is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close the window
        break

# Close all OpenCV windows after exiting the loop
cv2.destroyAllWindows()
