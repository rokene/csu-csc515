#!/usr/bin/env python3

import cv2
import os

# Load and display the image
image = cv2.imread('brain.jpg')
cv2.imshow('Brain Image', image)

# Save a copy of the image to the home directory
home_directory = os.path.expanduser("~")
save_path = os.path.join(home_directory, "brain_image_copy.jpg")
cv2.imwrite(save_path, image)
print(f"Image saved to {save_path}")

# Wait until the window is closed manually
cv2.waitKey(0)

# Close all OpenCV windows (for cleanup)
cv2.destroyAllWindows()
