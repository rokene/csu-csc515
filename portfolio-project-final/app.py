#!/usr/bin/env python3
"""
Full pipeline to:
  - Detect license plates with fallback scaling
  - Rectify (deskew) each plate
  - Classify plate as Russian vs. Non-Russian by identifying a separate box on the right
    which typically contains the 2-3 digit region code
  - Validate the split by checking if the right box contains 2-3 digits via OCR
  - Run region-specific OCR
  - Display results (annotated original + each plate)
"""

import os
import glob
import math
import logging
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import argparse

# -------------------------------------------------------------------
# CONFIG CONSTANTS
# -------------------------------------------------------------------

# Fallback scale factors for detection (try original, 1.5x, 2x, 3x, 4x)
SCALE_FACTORS = [1.0, 1.5, 2.0, 3.0, 4.0]

# Display in a grid with up to 5 columns
MAX_COLUMNS = 5

# If plate’s min dimension < 50 px, we treat it as "small"
SMALL_PLATE_DIM_THRESHOLD = 50

# Region-specific Tesseract whitelists
RUSSIAN_WHITELIST = "АВЕКМНОРСТУХ0123456789"  # Cyrillic + digits
NON_RUSSIAN_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Region-specific OCR configs
RUSSIAN_OCR_CONFIG = (
    "--oem 3 --psm 8 "
    # f"-c tessedit_char_whitelist={RUSSIAN_WHITELIST}"
)

NON_RUSSIAN_OCR_CONFIG = (
    "--oem 3 --psm 7 "
    f"-c tessedit_char_whitelist={NON_RUSSIAN_WHITELIST}"
)

# -------------------------------------------------------------------
# LOGGING CONFIGURATION
# -------------------------------------------------------------------
def setup_logging(debug: bool = False):
    """
    Sets up logging configuration.
    
    Args:
        debug (bool): If True, set logging level to DEBUG, else INFO.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Avoid adding multiple handlers if already present
    if not logger.handlers:
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # Add formatter to handler
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)

# -------------------------------------------------------------------
# DETECTION: Fallback Scaling
# -------------------------------------------------------------------
def detect_plates_with_fallback_scaling(
    img_bgr: np.ndarray, 
    cascade_path: str,
    scale_factors: List[float] = SCALE_FACTORS
) -> List[Tuple[int,int,int,int]]:
    """
    Attempts plate detection at multiple scales.
    Returns bounding boxes (x,y,w,h) in original coords or empty list if none found.
    """
    for sf in scale_factors:
        if sf == 1.0:
            scaled_bgr = img_bgr
        else:
            scaled_bgr = cv2.resize(
                img_bgr, None,
                fx=sf, fy=sf,
                interpolation=cv2.INTER_CUBIC
            )

        scaled_gray = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2GRAY)
        scaled_gray_eq = cv2.equalizeHist(scaled_gray)
        scaled_blur = cv2.GaussianBlur(scaled_gray_eq, (3, 3), 0)

        plate_cascade = cv2.CascadeClassifier(cascade_path)
        detections_scaled = plate_cascade.detectMultiScale(
            scaled_blur,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(detections_scaled) > 0:
            logging.info(f"Detected {len(detections_scaled)} plate(s) at scale factor {sf}")
            # Convert coordinates back
            bboxes_original = []
            for (x_s, y_s, w_s, h_s) in detections_scaled:
                x_o = int(x_s / sf)
                y_o = int(y_s / sf)
                w_o = int(w_s / sf)
                h_o = int(h_s / sf)
                bboxes_original.append((x_o, y_o, w_o, h_o))
            return bboxes_original
        else:
            logging.info(f"No plates at scale factor {sf}, trying next...")

    logging.info("No plates detected after all fallback scales.")
    return []

# -------------------------------------------------------------------
# RECTIFICATION: Deskew Plate
# -------------------------------------------------------------------
def rectify_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Finds largest 4-corner contour to deskew the plate.
    Returns the warped (rectified) plate or the original if not found.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logging.info("No contours found for rectification.")
        return plate_bgr

    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.04 * peri, True)
    if len(approx) != 4:
        logging.info("Largest contour does not have 4 corners. Skipping rectification.")
        return plate_bgr

    pts = approx.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # top-left
    rect[0] = pts[np.argmin(s)]
    # bottom-right
    rect[2] = pts[np.argmax(s)]
    # top-right
    rect[1] = pts[np.argmin(diff)]
    # bottom-left
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(br - tr)
    heightB = np.linalg.norm(bl - tl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(plate_bgr, M, (maxWidth, maxHeight))

    logging.info("Plate rectified successfully.")
    return warped

# -------------------------------------------------------------------
# SPLIT BY VERTICAL LINE WITH DEBUGGING AND STRICT VERTICALITY
# -------------------------------------------------------------------
def split_plate_by_vertical_line(
    plate_bgr: np.ndarray,
    debug: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Attempts to find a vertical black line in the plate image to split it into left and right regions.
    Returns (left_bgr, right_bgr) if a valid split is found, else (None, None).
    
    Args:
        plate_bgr (np.ndarray): The BGR image of the license plate.
        debug (bool): If True, displays intermediate images for debugging purposes.
    
    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Left and right split images or (None, None).
    """
    logging.info("Starting split_plate_by_vertical_line function")

    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    logging.info("Converted plate to grayscale")

    # Step 2: Apply CLAHE for Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    logging.info("Applied CLAHE for contrast enhancement")

    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(gray_clahe, cmap='gray')
        plt.title("Contrast Enhanced Grayscale Image (CLAHE)")
        plt.axis('off')
        plt.show()

    # Step 3: Apply Adaptive Thresholding with Inversion
    thresh = cv2.adaptiveThreshold(
        gray_clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Invert to make lines white
        blockSize=15,
        C=10
    )
    logging.info("Applied adaptive thresholding with THRESH_BINARY_INV")

    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(thresh, cmap='gray')
        plt.title("Adaptive Thresholded & Inverted Image")
        plt.axis('off')
        plt.show()

    # Morphological Operations to Enhance Vertical Lines
    # Adjusted kernel size for taller vertical lines and increased iterations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))  # Increased height
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)  # Increased iterations
    logging.info("Performed morphological opening with adjusted kernel to enhance vertical lines")
    
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(detected_lines, cmap='gray')
        plt.title("Morphologically Processed Image (Vertical Lines Enhanced)")
        plt.axis('off')
        plt.show()

    # Detect Lines Using Hough Transform
    plate_height = plate_bgr.shape[0]
    min_line_length = int(0.90 * plate_height)  # Reduced to 70% of plate height
    lines = cv2.HoughLinesP(
        detected_lines,
        rho=1,
        theta=np.pi / 180,
        threshold=20,                # Reduced threshold
        minLineLength=min_line_length,  # Reduced minLineLength
        maxLineGap=30                 # Increased maxLineGap
    )

    if lines is not None:
        logging.info(f"Detected {len(lines)} lines using HoughLinesP")
    else:
        logging.info("No lines detected using HoughLinesP")

    # Initialize list to hold valid split columns
    valid_split_columns = []
    search_zone_start = int(0.5 * plate_bgr.shape[1])  # Start searching from the middle
    search_zone_end = plate_bgr.shape[1]

    debug_image = plate_bgr.copy()  # For visualization of detected lines

    if lines is not None:
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            line_length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            logging.info(f"Line {idx}: ({x1}, {y1}) to ({x2}, {y2}), angle: {angle:.2f} degrees, length: {line_length:.2f} pixels")

            # Check if the line is strictly vertical by defining a tighter angle range
            # For "pretty vertical" lines, consider angles close to 90 degrees
            # Adjust the range as needed (e.g., 85 to 95 degrees)
            if 85 < abs(angle) < 95 and min(x1, x2) >= search_zone_start:
                # Check if the line length covers at least 70% of the plate's height
                if line_length >= 0.7 * plate_height:
                    split_col = int((x1 + x2) / 2)
                    valid_split_columns.append(split_col)
                    logging.info(f"Valid vertical line detected at column: {split_col} with length: {line_length:.2f}")
                    cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green line for valid lines
                else:
                    logging.info(f"Ignored line {idx} due to insufficient length: {line_length:.2f} pixels")
            else:
                logging.info(f"Ignored line {idx} due to angle: {angle:.2f} degrees or position: {min(x1, x2)} < {search_zone_start}")

    if debug and lines is not None:
        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Valid Vertical Lines")
        plt.axis('off')
        plt.show()

    # Select the split column if any valid lines were detected
    if valid_split_columns:
        split_col = max(valid_split_columns)  # Choose the rightmost valid vertical line
        logging.info(f"Selected split column at position: {split_col} based on valid vertical lines")
    
        # Validate split column position to ensure it's not too close to the edges
        W = plate_bgr.shape[1]
        min_distance = int(0.05 * W)  # 5% of plate width as minimum distance from edges
    
        if min_distance < split_col < W - min_distance:
            left_bgr = plate_bgr[:, :split_col]
            right_bgr = plate_bgr[:, split_col:]
            logging.info(f"Valid split found via Hough Transform at column: {split_col}")
            
            if debug:
                # Visualize the split plates
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB))
                plt.title("Left Split")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB))
                plt.title("Right Split")
                plt.axis("off")
                plt.show()
    
            return left_bgr, right_bgr
        else:
            logging.warning(f"Split column at {split_col} is too close to the edges (Plate width: {W})")
    
    # If no valid split lines were detected, do not split
    logging.info("No valid vertical split lines detected. Skipping split.")
    return None, None

# -------------------------------------------------------------------
# ADVANCED PREPROCESSING FOR OCR
# -------------------------------------------------------------------
def advanced_preprocessing_for_ocr(plate_bgr: np.ndarray) -> np.ndarray:
    """
    - If small, upscale
    - Grayscale, CLAHE, morphological close
    - Adaptive threshold
    Returns binarized image for Tesseract.
    """
    h, w = plate_bgr.shape[:2]
    if min(h, w) < SMALL_PLATE_DIM_THRESHOLD:
        plate_bgr = cv2.resize(plate_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        logging.info("Upscaled small plate for better OCR accuracy.")

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(gray_clahe, cv2.MORPH_CLOSE, kernel_close)

    bin_img = cv2.adaptiveThreshold(
        closed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=5
    )
    logging.info("Applied advanced preprocessing for OCR.")
    return bin_img

def advanced_preprocessing_for_ocr_rus_left(plate_bgr: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Advanced preprocessing to prepare the license plate image for OCR.
    Enhancements:
        - Upscale if the plate is small.
        - Convert to grayscale.
        - Apply CLAHE for contrast enhancement.
        - Morphological closing to bridge gaps in characters.
        - Adaptive thresholding for binarization.
        - Noise removal using morphological operations.
        - Contour detection and filtering to isolate valid characters.
        - Optionally, crop the image to focus on character regions.
    
    Args:
        plate_bgr (np.ndarray): The BGR image of the license plate.
        debug (bool): If True, displays intermediate images for debugging purposes.
    
    Returns:
        np.ndarray: The preprocessed binary image ready for OCR.
    """
    logging.info("Starting advanced preprocessing for OCR.")

    # Step 1: Upscale if the plate is small
    h, w = plate_bgr.shape[:2]
    if min(h, w) < SMALL_PLATE_DIM_THRESHOLD:
        plate_bgr = cv2.resize(plate_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        logging.info("Upscaled small plate for better OCR accuracy.")
        if debug:
            plt.figure(figsize=(6, 4))
            plt.imshow(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
            plt.title("Upscaled Plate")
            plt.axis('off')
            plt.show()

    # Step 2: Convert to Grayscale
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.show()

    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(gray_clahe, cmap='gray')
        plt.title("CLAHE Enhanced Image")
        plt.axis('off')
        plt.show()

    # Step 4: Morphological Closing to Bridge Gaps in Characters
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(gray_clahe, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(closed, cmap='gray')
        plt.title("Morphological Closing")
        plt.axis('off')
        plt.show()

    # Step 5: Adaptive Thresholding for Binarization
    bin_img = cv2.adaptiveThreshold(
        closed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=5
    )
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(bin_img, cmap='gray')
        plt.title("Adaptive Thresholding")
        plt.axis('off')
        plt.show()

    # Step 6: Noise Removal using Morphological Opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_open, iterations=1)
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(opened, cmap='gray')
        plt.title("Morphological Opening (Noise Removal)")
        plt.axis('off')
        plt.show()

    # Step 7: Contour Detection to Isolate Valid Characters
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)

        # Define criteria for valid character contours
        # Adjust these thresholds based on your specific dataset
        if 0.2 < aspect_ratio < 1.0 and 100 < area < 1000:
            valid_contours.append((x, y, w, h))
            if debug:
                cv2.rectangle(plate_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if debug:
        debug_image = plate_bgr.copy()
        for (x, y, w, h) in valid_contours:
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        plt.title("Valid Contours Highlighted")
        plt.axis('off')
        plt.show()

    # Step 8: Create a Mask of Valid Characters
    mask = np.zeros_like(opened)
    for (x, y, w, h) in valid_contours:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Apply the mask to the binarized image
    masked_bin = cv2.bitwise_and(opened, opened, mask=mask)
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(masked_bin, cmap='gray')
        plt.title("Masked Binarized Image")
        plt.axis('off')
        plt.show()

    # Optional Step 9: Crop the Image to the Region of Valid Characters
    # This step focuses OCR on the area containing valid characters
    # Compute bounding boxes of valid contours to determine the cropping region
    if valid_contours:
        x_coords = [x for (x, y, w, h) in valid_contours]
        y_coords = [y for (x, y, w, h) in valid_contours]
        w_coords = [w for (x, y, w, h) in valid_contours]
        h_coords = [h for (x, y, w, h) in valid_contours]

        x_min = max(min(x_coords) - 5, 0)
        y_min = max(min(y_coords) - 5, 0)
        x_max = min(max([x + w for (x, y, w, h) in valid_contours]) + 5, plate_bgr.shape[1])
        y_max = min(max([y + h for (x, y, w, h) in valid_contours]) + 5, plate_bgr.shape[0])

        cropped_bin = masked_bin[y_min:y_max, x_min:x_max]
        if debug:
            plt.figure(figsize=(6, 4))
            plt.imshow(cropped_bin, cmap='gray')
            plt.title("Cropped Binarized Image (Valid Characters)")
            plt.axis('off')
            plt.show()
    else:
        logging.warning("No valid contours found. Proceeding with the full binarized image.")
        cropped_bin = opened  # Fallback to the full image if no valid contours are found

    # Step 10: Final Noise Removal (Optional)
    # Remove any remaining small noise
    kernel_final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final_clean = cv2.morphologyEx(cropped_bin, cv2.MORPH_OPEN, kernel_final, iterations=1)
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(final_clean, cmap='gray')
        plt.title("Final Cleaned Image")
        plt.axis('off')
        plt.show()

    logging.info("Advanced preprocessing for OCR completed.")
    return final_clean


# -------------------------------------------------------------------
# OCR FUNCTIONS
# -------------------------------------------------------------------
def do_ocr_russian_left(plate_bgr: np.ndarray) -> str:
    """
    Advanced preprocessing + Tesseract with Russian whitelist/config.
    """
    bin_img = advanced_preprocessing_for_ocr_rus_left(plate_bgr)
    bin_img = cv2.resize(bin_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(bin_img, lang="rus", config=RUSSIAN_OCR_CONFIG)
    logging.info(f"OCR Result (Russian): {text.strip()}")
    return text.strip()

def do_ocr_nonrussian(plate_bgr: np.ndarray) -> str:
    """
    Advanced preprocessing + Tesseract with Non-Russian (e.g., eng) config.
    """
    bin_img = advanced_preprocessing_for_ocr(plate_bgr)
    bin_img = cv2.resize(bin_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(bin_img, lang="eng", config=NON_RUSSIAN_OCR_CONFIG)
    logging.info(f"OCR Result (Non-Russian): {text.strip()}")
    return text.strip()

def do_ocr_russian_right(right_bgr: np.ndarray, debug: bool = False) -> str:
    """
    Preprocess the right split and perform OCR to extract the region code.

    Args:
        right_bgr (np.ndarray): The BGR image of the right split.
        debug (bool): If True, enables debugging visualizations.

    Returns:
        str: The recognized region code.
    """
    logging.info("Starting OCR for the right split (Region Code).")
    
    # Preprocess the right split to isolate digits
    preprocessed = preprocess_right_bgr(right_bgr, debug=debug)

    # Resize for better OCR accuracy
    resized = cv2.resize(preprocessed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(resized, cmap='gray')
        plt.title("Right Split - Preprocessed for OCR")
        plt.axis('off')
        plt.show()

    # Perform OCR with digit-only whitelist
    region_code = pytesseract.image_to_string(
        resized,
        lang="rus",
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    )

    region_code_clean = ''.join(filter(str.isdigit, region_code.strip()))
    logging.info(f"OCR Result for Region Code: {region_code_clean}")

    return region_code_clean


# -------------------------------------------------------------------
# CLASSIFICATION: RUS vs NON-RUS By Vertical Split and OCR Verification
# -------------------------------------------------------------------
def classify_plate_by_vertical_split(
    plate_bgr: np.ndarray,
    debug: bool = False
) -> Tuple[str, Optional[str], Optional[np.ndarray]]:
    """
    Classifies the plate as "Russian" if a vertical split is detected and the right
    split contains 2-3 digits. Otherwise, classifies as "Non-Russian."

    Args:
        plate_bgr (np.ndarray): The BGR image of the license plate.
        debug (bool): If True, enables debugging visualizations.

    Returns:
        Tuple[str, Optional[str], Optional[np.ndarray]]:
            - Classification label ("Russian" or "Non-Russian")
            - Recognized text from the right split (if Russian) or None
            - Left split image (if Russian) or None
    """
    left_bgr, right_bgr = split_plate_by_vertical_line(plate_bgr, debug=debug)

    if left_bgr is not None and right_bgr is not None:
        return "Russian", left_bgr, right_bgr
        # # Perform OCR on the right split to verify it contains 2-3 digits
        # right_text = do_ocr_russian_left(right_bgr)
        # right_text_clean = ''.join(filter(str.isdigit, right_text.strip()))

        # if 2 <= len(right_text_clean) <= 3:
        #     logging.info(f"Plate classified as Russian with region code: {right_text_clean}")
        #     return "Russian", right_text_clean, left_bgr
        # else:
        #     logging.info("Split detected but region code does not contain 2-3 digits.", right_text_clean)
    
    # If no valid split or OCR verification failed, classify as Non-Russian
    logging.info("Plate classified as Non-Russian.")
    return "Non-Russian", None, None

def preprocess_right_bgr(right_bgr: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Preprocess the right side of the split plate to isolate and enhance the numerical region
    by cropping off the bottom part containing "RUS" and the flag, and eliminating artifacts.

    Args:
        right_bgr (np.ndarray): The BGR image of the right split.
        debug (bool): If True, displays intermediate images for debugging purposes.

    Returns:
        np.ndarray: The preprocessed image ready for OCR.
    """
    logging.info("Starting preprocessing for right_bgr to isolate numerical region.")

    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(gray, cmap='gray')
        plt.title("Right Split - Grayscale")
        plt.axis('off')
        plt.show()

    # Step 2: Apply Morphological Opening to Remove Noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_open, iterations=1)
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(opened, cmap='gray')
        plt.title("Right Split - Morphological Opening")
        plt.axis('off')
        plt.show()

    # Step 3: Apply Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        opened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10
    )
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(thresh, cmap='gray')
        plt.title("Right Split - Adaptive Thresholding")
        plt.axis('off')
        plt.show()

    # Step 4: Morphological Closing to Bridge Gaps Within Digits
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(closed, cmap='gray')
        plt.title("Right Split - Morphological Closing")
        plt.axis('off')
        plt.show()

    # Step 5: Calculate Horizontal Projection
    horizontal_sum = np.sum(closed, axis=1)
    if debug:
        plt.figure(figsize=(10, 4))
        plt.plot(horizontal_sum)
        plt.title("Horizontal Projection")
        plt.xlabel("Row")
        plt.ylabel("Sum of White Pixels")
        plt.show()

    # Step 6: Identify the Whitespace Row
    # Define a threshold to detect significant whitespace
    # For example, rows with sum > 70% of the maximum sum are considered as whitespace
    threshold_ratio = 0.7
    max_sum = np.max(horizontal_sum)
    whitespace_threshold = threshold_ratio * max_sum

    # Find all rows that exceed the threshold
    potential_whitespace_rows = np.where(horizontal_sum >= whitespace_threshold)[0]

    if len(potential_whitespace_rows) == 0:
        logging.warning("No significant whitespace detected. Proceeding with default cropping.")
        whitespace_row = right_bgr.shape[0] // 2  # Default to middle if no whitespace found
    else:
        # Assume the first occurrence is the separator
        whitespace_row = potential_whitespace_rows[0]
        logging.info(f"Detected significant whitespace at row: {whitespace_row}")

    # Optional: Visualize the detected whitespace
    if debug:
        debug_image = right_bgr.copy()
        cv2.line(debug_image, (0, whitespace_row), (right_bgr.shape[1], whitespace_row), (0, 255, 0), 2)
        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Whitespace Line")
        plt.axis('off')
        plt.show()

    # Step 7: Crop the Image Above the Whitespace
    # Add a small margin to ensure all digits are included
    margin = 5  # pixels
    if whitespace_row - margin <= 0:
        logging.warning("Calculated whitespace row is too close to the top. Adjusting margin.")
        cropped_thresh = closed[:whitespace_row, :]
    else:
        cropped_thresh = closed[:whitespace_row - margin, :]

    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(cropped_thresh, cmap='gray')
        plt.title("Right Split - Cropped Numerical Region")
        plt.axis('off')
        plt.show()

    # Step 8: Remove Small Artifacts Using Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cropped_thresh, connectivity=8)
    min_area = 50  # Minimum area to be considered a valid digit (adjust as needed)
    cleaned = np.zeros_like(cropped_thresh)

    for i in range(1, num_labels):  # Skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    if debug:
        plt.figure(figsize=(6, 4))
        plt.imshow(cleaned, cmap='gray')
        plt.title("Right Split - Cleaned Numerical Region")
        plt.axis('off')
        plt.show()

    return cleaned


# -------------------------------------------------------------------
# MAIN: Annotate, Classify, and OCR
# -------------------------------------------------------------------
def annotate_crop_classify_ocr(
    img_bgr: np.ndarray,
    bboxes: List[Tuple[int,int,int,int]],
    debug: bool = False
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, str, str]]]:
    """
    - Draw bounding boxes on the original image
    - For each box:
        - Crop and rectify
        - Classify as "Russian" or "Non-Russian" based on vertical split and OCR verification
        - Perform region-specific OCR
    Args:
        img_bgr (np.ndarray): The original BGR image.
        bboxes (List[Tuple[int, int, int, int]]): List of bounding boxes (x, y, w, h).
        debug (bool): If True, enables debugging visualizations.

    Returns:
        Tuple[np.ndarray, List[Tuple[np.ndarray, str, str]]]: 
            - Annotated image
            - List of tuples containing (rectified_plate, label, recognized_text)
    """
    annotated = img_bgr.copy()
    plate_info_list = []

    for (x, y, w, h) in bboxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        plate_crop = img_bgr[y:y+h, x:x+w]

        # Rectify
        rectified_plate = rectify_plate(plate_crop)

        # Classify
        label, left_bgr, right_bgr = classify_plate_by_vertical_split(rectified_plate, debug=debug)

        # OCR
        if label == "Russian" and left_bgr is not None:
            left_text = do_ocr_russian_left(left_bgr)
            right_text = do_ocr_russian_right(right_bgr, debug=debug)
            recognized_text = f"Plate: {left_text} Region: {right_text}"
            logging.info(f"OCR Result for Russian plate: {recognized_text}")
        else:
            recognized_text = do_ocr_nonrussian(rectified_plate)
            logging.info(f"OCR Result for Non-Russian plate: {recognized_text}")

        plate_info_list.append((rectified_plate, label, recognized_text))

    return annotated, plate_info_list

# -------------------------------------------------------------------
# DISPLAY
# -------------------------------------------------------------------
def show_results(
    annotated_img_bgr: np.ndarray,
    plate_info_list: List[Tuple[np.ndarray, str, str]],
    title: str
):
    """
    Displays a single figure:
      - First subplot: annotated original image
      - Subsequent subplots: rectified plates with classification and OCR results
    """
    annotated_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)

    total_images = 1 + len(plate_info_list)
    rows = math.ceil(total_images / MAX_COLUMNS)

    fig, axs = plt.subplots(rows, MAX_COLUMNS, figsize=(5 * MAX_COLUMNS, 4 * rows))
    fig.suptitle(title, fontsize=16)

    # Handle case when rows == 1 and MAX_COLUMNS ==1 to make axs iterable
    if rows == 1 and MAX_COLUMNS == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.expand_dims(axs, axis=0)
    elif MAX_COLUMNS ==1:
        axs = np.expand_dims(axs, axis=1)
    else:
        axs = np.array(axs).reshape(rows, MAX_COLUMNS)

    # Original image in first subplot
    axs[0, 0].imshow(annotated_rgb)
    axs[0, 0].set_title("Original (Annotated)", fontsize=10)
    axs[0, 0].axis("off")

    # Plates
    index = 1
    for (plate_bgr, label, text) in plate_info_list:
        r = index // MAX_COLUMNS
        c = index % MAX_COLUMNS
        if r >= rows:
            # Prevent index out of range in case of more images than subplots
            break
        plate_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        axs[r, c].imshow(plate_rgb)
        if label == "Russian" and "Region" in text:
            axs[r, c].set_title(f"{label}: {text}", fontsize=10)
        else:
            axs[r, c].set_title(f"{label}: {text}", fontsize=10)
        axs[r, c].axis("off")
        index += 1

    # Hide any unused subplots
    while index < rows * MAX_COLUMNS:
        r = index // MAX_COLUMNS
        c = index % MAX_COLUMNS
        axs[r, c].axis("off")
        index += 1

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# PROCESS A SINGLE IMAGE
# -------------------------------------------------------------------
def process_image(
    img_path: str,
    cascade_path: str,
    debug: bool = False
):
    """
    Processes a single image:
      - Load image
      - Detect plates with fallback scaling
      - Annotate, classify, and OCR
      - Display results
    Args:
        img_path (str): Path to the image file.
        cascade_path (str): Path to the Haar cascade XML file for plate detection.
        debug (bool): If True, enables debugging visualizations.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        logging.error(f"Could not load {img_path}")
        return

    bboxes = detect_plates_with_fallback_scaling(img_bgr, cascade_path)
    if not bboxes:
        logging.info(f"No plates detected in {img_path}")
    else:
        logging.info(f"Detected {len(bboxes)} plate(s) in {img_path}")

    annotated, plate_info_list = annotate_crop_classify_ocr(img_bgr, bboxes, debug=debug)
    show_results(annotated, plate_info_list, os.path.basename(img_path))

# -------------------------------------------------------------------
# MAIN: Process Directory
# -------------------------------------------------------------------
def main(
    images_dir: str = "images",
    cascade_path: str = "data/haarcascade_russian_plate_number.xml",
    debug: bool = False
):
    """
    Processes all images in 'images_dir':
      - Detect plates
      - Classify and OCR
      - Display results
    Args:
        images_dir (str): Directory containing images to process.
        cascade_path (str): Path to the Haar cascade XML file for plate detection.
        debug (bool): If True, enables debugging visualizations.
    """
    image_paths = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    if not image_paths:
        logging.warning(f"No images found in '{images_dir}'")
        return

    image_paths.sort()

    for img_path in image_paths:
        logging.info(f"Processing: {img_path}")
        process_image(img_path, cascade_path, debug=debug)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="License Plate Processing Pipeline with Debugging")
    parser.add_argument(
        "--images_dir",
        type=str,
        default="images",
        help="Directory containing images to process."
    )
    parser.add_argument(
        "--cascade_path",
        type=str,
        default="data/haarcascade_russian_plate_number.xml",
        help="Path to the Haar cascade XML file for plate detection."
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Enable debugging visualizations and DEBUG level logging."
    )

    args = parser.parse_args()

    # Configure logging
    setup_logging(debug=args.debug)

    # Run main function with provided arguments
    main(
        images_dir=args.images_dir,
        cascade_path=args.cascade_path,
        debug=args.debug
    )
