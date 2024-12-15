import cv2
import numpy as np
import os

def convert_to_bgr(image):
    """Convert a single-channel image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def add_title(image, title):
    """Add a title to the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    color = (0, 0, 255)  # Red color in BGR
    # Calculate the position for the text
    text_size, _ = cv2.getTextSize(title, font, font_scale, thickness)
    text_x = 10
    text_y = text_size[1] + 10
    cv2.putText(image, title, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def resize_image(image, width):
    """Resize image to a given width while maintaining aspect ratio."""
    aspect_ratio = image.shape[1] / image.shape[0]
    height = int(width / aspect_ratio)
    resized = cv2.resize(image, (width, height))
    return resized

# Load the image
image = cv2.imread('data/hey.png')

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise before thresholding
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
binary = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# Define a rectangular structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Apply dilation
dilated = cv2.dilate(binary, kernel, iterations=1)

# Apply erosion
eroded = cv2.erode(binary, kernel, iterations=1)

# Apply opening
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# Apply closing
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

# List of images and their corresponding titles
images = [
    (image, 'Original Image'),
    (gray, 'Grayscale'),
    (binary, 'Binary Image'),
    (dilated, 'Dilated'),
    (eroded, 'Eroded'),
    (opened, 'Opened'),
    (closed, 'Closed')
]

# Convert all images to BGR and add titles
processed_images = []
for img, title in images:
    if len(img.shape) == 2:  # If the image is single-channel
        img_bgr = convert_to_bgr(img)
    else:
        img_bgr = img.copy()
    img_with_title = add_title(img_bgr, title)
    processed_images.append(img_with_title)

# Define desired width for each image
desired_width = 700

# Resize all processed images
resized_images = [resize_image(img, desired_width) for img in processed_images]

# Calculate grid dimensions
rows = 2
cols = 4

# Initialize an empty list to hold rows of images
grid_rows = []

# Iterate through images and arrange them into rows
for i in range(rows):
    row_images = []
    for j in range(cols):
        idx = i * cols + j
        if idx < len(resized_images):
            row_images.append(resized_images[idx])
        else:
            # If no image is available, add a blank image
            blank_image = np.ones_like(resized_images[0], dtype=np.uint8) * 255  # White image
            row_images.append(blank_image)
    # Concatenate images horizontally to form a row
    row = cv2.hconcat(row_images)
    grid_rows.append(row)

# Concatenate all rows vertically to form the final grid
grid_image = cv2.vconcat(grid_rows)

# Display the grid
cv2.imshow('Morphological Operations Grid', grid_image)

# Display each image in separate windows
window_names = [
    'Original Image',
    'Grayscale',
    'Binary Image',
    'Dilated',
    'Eroded',
    'Opened',
    'Closed'
]

for img, title in zip(resized_images, window_names):
    cv2.imshow(title, img)

# Create a directory to save processed images
output_dir = 'processed_images'
os.makedirs(output_dir, exist_ok=True)

# Define filenames for each processed image
filenames = [
    'original_image.jpg',
    'grayscale.jpg',
    'binary_image.jpg',
    'dilated.jpg',
    'eroded.jpg',
    'opened.jpg',
    'closed.jpg'
]

# Save each image
for img, filename in zip(processed_images, filenames):
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, img)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
