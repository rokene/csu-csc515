#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('puppy.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB

# Extract channels
blue_channel, green_channel, red_channel = cv2.split(image)

# Merge channels to reconstruct the image
original_image = cv2.merge((blue_channel, green_channel, red_channel))

# Swap red and green channels (GRB)
swapped_image = cv2.merge((blue_channel, red_channel, green_channel))

# Plot the results
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(red_channel, cmap="Reds")
axes[1].set_title("Red Channel")
axes[1].axis("off")

axes[2].imshow(green_channel, cmap="Greens")
axes[2].set_title("Green Channel")
axes[2].axis("off")

axes[3].imshow(blue_channel, cmap="Blues")
axes[3].set_title("Blue Channel")
axes[3].axis("off")

plt.show()

# Plot reconstructed images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_image)
axes[0].set_title("Reconstructed Image")
axes[0].axis("off")

axes[1].imshow(swapped_image)
axes[1].set_title("Swapped Image (GRB)")
axes[1].axis("off")

plt.show()
