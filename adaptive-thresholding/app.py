#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt

def show_images(images, titles, rows=1, cols=3, figsize=(15, 5), image_name=""):
    """
    Utility function to display multiple images in a grid using Matplotlib.
    images: list of images (NumPy arrays)
    titles: list of titles (strings)
    """
    fig = plt.figure(figsize=figsize)
    fig.canvas.manager.set_window_title(image_name + ": Press any key to continue.")
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        # If the image is 2D, assume it's grayscale
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            # If it has 3 dims, assume BGR color -> convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def adaptive_threshold_demo(image_path, image_name, block_size=11, C=2, blur_ksize=(5,5)):
    """
    Loads an image from image_path, converts it to grayscale,
    and applies two types of adaptive thresholding. Also compares
    results with and without Gaussian blurring.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open or find the image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding (No Blur) Adaptive Mean
    thresh_mean = cv2.adaptiveThreshold(
        gray,                  # source image
        255,                   # max value
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,            # blockSize
        C                      # constant subtracted
    )

    # Adaptive Gaussian
    thresh_gaussian = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # Apply Gaussian Blurring
    gray_blur = cv2.GaussianBlur(gray, blur_ksize, 0)

    # Apply adaptive thresholding (With Blur) Adaptive Mean (blurred)
    thresh_mean_blur = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # Adaptive Gaussian (blurred)
    thresh_gaussian_blur = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # Display two rows:
    #    Row 1 -> Original color, Mean (no blur), Gaussian (no blur)
    #    Row 2 -> Blurred Gray, Mean (blurred), Gaussian (blurred)
    images_to_show = [
        img, 
        thresh_mean, 
        thresh_gaussian,
        gray_blur,
        thresh_mean_blur,
        thresh_gaussian_blur
    ]

    titles = [
        "Original (Color)",
        "Adaptive Mean",
        "Adaptive Gaussian",
        "Blurred Gray",
        "Adaptive Mean (Blurred)",
        "Adaptive Gaussian (Blurred)"
    ]

    show_images(images_to_show, titles, rows=2, cols=3, figsize=(18, 10), image_name=image_name)

# -------------------------------------------------------------------
# Main Execution: Provide three images (indoor, outdoor, close-up)
# -------------------------------------------------------------------

# Indoor Scene
print("Processing indoor scene...")
adaptive_threshold_demo("data/indoor.jpg", "Indoor", block_size=11, C=2)

# Outdoor Scenery
print("Processing outdoor scenery...")
adaptive_threshold_demo("data/outdoor.jpg", "Outdoor", block_size=15, C=5)

# Close-up Scene of a Single Object
print("Processing close-up object...")
adaptive_threshold_demo("data/closeup.jpg", "Close-Up", block_size=11, C=5)
