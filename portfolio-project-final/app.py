import os
import glob
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Preprocessing function to prepare the plate image for Tesseract
###############################################################################
def preprocess_for_ocr(gray_img):
    # 1. Resize to make it bigger (2x or 3x)
    scale_factor = 2
    new_width = gray_img.shape[1] * scale_factor
    new_height = gray_img.shape[0] * scale_factor
    gray_resized = cv2.resize(gray_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # 2. Morphological closing to connect broken parts of characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(gray_resized, cv2.MORPH_CLOSE, kernel)

    # 3. Threshold (Otsu or Adaptive)
    # Option A: Otsu
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Option B: Adaptive
    # thresh = cv2.adaptiveThreshold(
    #     closed, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     11, 2
    # )

    return thresh

###############################################################################
# Function to detect and OCR the license plate in a single image
###############################################################################
def detect_and_ocr_license_plate(img_path, plate_cascade):
    """
    Returns:
      (detected_img, result_title) where
        - detected_img is the BGR image with bounding boxes drawn,
        - result_title is a string describing OCR results (or an error message).
    """
    # 1. Load and preprocess the full image
    img_color = cv2.imread(img_path)
    if img_color is None:
        return None, f"Error loading image: {img_path}"

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_gray_eq = cv2.equalizeHist(img_gray)
    img_blur = cv2.GaussianBlur(img_gray_eq, (3, 3), 0)

    # 2. Detect plates
    plates = plate_cascade.detectMultiScale(img_blur, scaleFactor=1.1, minNeighbors=5)

    # By default, assume we haven't found anything
    result_title = "No plate detected"

    # 3. For each detected plate, try OCR
    for (x, y, w, h) in plates:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Extract plate region from the original gray
        plate_region = img_gray[y : y + h, x : x + w]

        # Preprocess for OCR
        plate_bin = preprocess_for_ocr(plate_region)

        # --psm 6: Assume a uniform block of text.
        # --psm 7: Single text line
        # --psm 8: Single word.
        # --psm 11: sparse text
        tesseract_config = (
            "--psm 8 "
            # "--oem 3 "
            "-c tessedit_char_whitelist=ABEKMHOPCTYX0123456789"
        )

        recognized_text = pytesseract.image_to_string(
            plate_bin,
            lang='rus',  
            config=tesseract_config
        ).strip()

        if recognized_text:
            result_title = f"Plate Text: {recognized_text}"
        else:
            result_title = "Plate detected but no text recognized"

    return img_color, result_title

###############################################################################
# Main logic: process all images in "images" folder and display results
###############################################################################
def analyze_all_images():
    # Load your cascade (update path as needed)
    # Example: 'data/haarcascade_license_plate_rus_16stages.xml'
    plate_cascade = cv2.CascadeClassifier('data/haarcascade_license_plate_rus_16stages.xml')

    # Gather all images in the "images" directory
    # Adjust extensions as needed (jpg, png, jpeg, etc.)
    image_paths = glob.glob(os.path.join("images", "*.jpg"))
    image_paths += glob.glob(os.path.join("images", "*.png"))
    image_paths += glob.glob(os.path.join("images", "*.jpeg"))

    if not image_paths:
        print("No images found in the 'images' directory.")
        return

    # Sort the paths so they display in a consistent order
    image_paths.sort()

    # Create subplots: one row per image
    fig, axes = plt.subplots(nrows=len(image_paths), ncols=1, figsize=(10, 5 * len(image_paths)))

    # If there's only one image, axes won't be a list, so handle that case
    if len(image_paths) == 1:
        axes = [axes]  # make it a list for consistent iteration

    for ax, img_path in zip(axes, image_paths):
        detected_img, result_title = detect_and_ocr_license_plate(img_path, plate_cascade)

        if detected_img is not None:
            # Convert BGR to RGB for matplotlib
            rgb_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb_img)
            ax.set_title(f"{os.path.basename(img_path)} | {result_title}", fontsize=12)
        else:
            # If image loading failed
            ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))  # show a blank
            ax.set_title(result_title, fontsize=12)

        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_all_images()
