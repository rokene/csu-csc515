import cv2
import numpy as np

# Function to add labels to images
def add_label(image, label, position=(10, 30)):
    """
    Adds a label to the image.

    :param image: Grayscale image.
    :param label: Text label to add.
    :param position: Tuple indicating the position of the text.
    :return: Image with label.
    """
    labeled_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .7
    color = (0, 0, 255)
    thickness = 1
    cv2.putText(labeled_image, label, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return labeled_image

# Read the image in grayscale
image_path = 'Mod4CT1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image not found at the path: {image_path}")

# Define kernel sizes
kernel_sizes = [3, 5, 7]

# Define sigma values for Gaussian filter
sigma1 = 1.0
sigma2 = 2.0

# Dictionary to store filtered images
filtered_images = {}

for k in kernel_sizes:
    filtered_images[k] = {}
    
    # Mean Filter
    mean = cv2.blur(image, (k, k))
    filtered_images[k]['Mean'] = mean
    
    # Median Filter
    median = cv2.medianBlur(image, k)
    filtered_images[k]['Median'] = median
    
    # Gaussian Filter with sigma1
    gaussian1 = cv2.GaussianBlur(image, (k, k), sigma1)
    filtered_images[k]['Gaussian_Sigma1'] = gaussian1
    
    # Gaussian Filter with sigma2
    gaussian2 = cv2.GaussianBlur(image, (k, k), sigma2)
    filtered_images[k]['Gaussian_Sigma2'] = gaussian2

# List to hold rows of images
rows = []

for k in kernel_sizes:
    row_images = []
    
    # Mean Filter
    mean_label = f"Mean Filter {k}x{k}"
    mean_labeled = add_label(filtered_images[k]['Mean'], mean_label)
    row_images.append(mean_labeled)
    
    # Median Filter
    median_label = f"Median Filter {k}x{k}"
    median_labeled = add_label(filtered_images[k]['Median'], median_label)
    row_images.append(median_labeled)
    
    # Gaussian Filter Sigma1
    gaussian1_label = f"Gaussian σ={sigma1} {k}x{k}"
    gaussian1_labeled = add_label(filtered_images[k]['Gaussian_Sigma1'], gaussian1_label)
    row_images.append(gaussian1_labeled)
    
    # Gaussian Filter Sigma2
    gaussian2_label = f"Gaussian σ={sigma2} {k}x{k}"
    gaussian2_labeled = add_label(filtered_images[k]['Gaussian_Sigma2'], gaussian2_label)
    row_images.append(gaussian2_labeled)
    
    # Concatenate images horizontally for the current row
    row_concatenated = cv2.hconcat(row_images)
    rows.append(row_concatenated)

# Concatenate all rows vertically
grid_image = cv2.vconcat(rows)

# Display results in grid image
scale_percent = 125  # Percent of original size
width = int(grid_image.shape[1] * scale_percent / 100)
height = int(grid_image.shape[0] * scale_percent / 100)
dim = (width, height)
grid_image = cv2.resize(grid_image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('Filter Comparisons (3x4 Grid): Press any key to quit.', grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
