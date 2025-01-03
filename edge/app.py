import cv2
import numpy as np

def generate_synthetic_image(width=256, height=256,
                             background_intensity=50,
                             square_intensity=200,
                             circle_intensity=180):
    """
    Generate a synthetic image with a single square and a single circle.
    Returns:
      img (np.ndarray): Grayscale image (8-bit).
      square_coords (tuple): (x, y, side_length).
      circle_coords (tuple): (center_x, center_y, radius).
    """
    # Create uniform background
    img = np.full((height, width), background_intensity, dtype=np.uint8)
    
    # Define square parameters
    sq_x, sq_y = 50, 50
    sq_size = 40
    
    # Draw the filled-in square
    cv2.rectangle(img,
                  (sq_x, sq_y),
                  (sq_x + sq_size, sq_y + sq_size),
                  color=square_intensity,
                  thickness=-1)
    
    # Define circle parameters
    circle_center_x, circle_center_y = 150, 150
    circle_radius = 20
    
    # Draw the filled-in circle
    cv2.circle(img,
               (circle_center_x, circle_center_y),
               circle_radius,
               color=circle_intensity,
               thickness=-1)
    
    return img, (sq_x, sq_y, sq_size), (circle_center_x, circle_center_y, circle_radius)


def generate_ground_truth_mask(width, height, square_coords, circle_coords):
    """
    Generate a ground-truth mask (binary) of the edges of the square and circle.
    Returns:
      gt_mask (np.ndarray): Binary mask with 255 on true edges, 0 otherwise.
    """
    gt_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Unpack square coords
    sq_x, sq_y, sq_size = square_coords
    # Unpack circle coords
    cx, cy, r = circle_coords
    
    # Mark square boundary
    # Top edge
    gt_mask[sq_y, sq_x : sq_x + sq_size] = 255
    # Bottom edge
    gt_mask[sq_y + sq_size - 1, sq_x : sq_x + sq_size] = 255
    # Left edge
    gt_mask[sq_y : sq_y + sq_size, sq_x] = 255
    # Right edge
    gt_mask[sq_y : sq_y + sq_size, sq_x + sq_size - 1] = 255
    
    # Mark circle boundary (draw a circle with thickness=1 on an empty mask)
    circle_mask = np.zeros_like(gt_mask)
    cv2.circle(circle_mask, (cx, cy), r, 255, thickness=1)
    
    # Combine
    gt_mask = cv2.bitwise_or(gt_mask, circle_mask)
    
    return gt_mask


def add_gaussian_noise(img, sigma=20):
    """
    Add Gaussian noise of standard deviation sigma to an 8-bit grayscale image.
    Returns:
      noisy_img (np.ndarray): Noisy image (8-bit).
    """
    noise = np.random.normal(0, sigma, img.shape)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def edge_detection_canny(img, threshold1, threshold2):
    return cv2.Canny(img, threshold1, threshold2)


def edge_detection_sobel(img, threshold):
    """
    Apply Sobel in X and Y, compute gradient magnitude, then threshold.
    Returns a binary edge map.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mag = cv2.convertScaleAbs(mag)
    
    _, bin_edges = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
    return bin_edges


def edge_detection_laplacian(img, threshold):
    """
    Apply Laplacian operator, then threshold the absolute response.
    Returns a binary edge map.
    """
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    lap_abs = cv2.convertScaleAbs(lap)
    _, bin_edges = cv2.threshold(lap_abs, threshold, 255, cv2.THRESH_BINARY)
    return bin_edges


def evaluate_performance(pred_mask, gt_mask):
    """
    Evaluate performance using Precision, Recall, and F1-score.
    pred_mask, gt_mask: binary images (0 or 255) of the same shape.
    Returns:
      (precision, recall, f1): floats in [0, 1].
    """
    pred = pred_mask > 0
    gt = gt_mask > 0
    
    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, np.logical_not(gt)).sum()
    FN = np.logical_and(np.logical_not(pred), gt).sum()
    
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1


def display_side_by_side(title, img_list, scale_factor=1.0):
    """
    Display multiple images side by side in a single window.
    All images are converted to BGR for visualization.
    Args:
        title (str): The window name.
        img_list (list[np.ndarray]): List of images (grayscale or BGR).
        scale_factor (float): Optional scaling factor for display size.
    """
    bgr_images = []
    
    for img in img_list:
        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Optionally resize if you want them bigger/smaller
        if scale_factor != 1.0:
            w = int(img.shape[1] * scale_factor)
            h = int(img.shape[0] * scale_factor)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        bgr_images.append(img)
    
    # Concatenate horizontally
    concat_image = cv2.hconcat(bgr_images)
    cv2.imshow(title, concat_image)


def main():
    # ---------------------------------------------------------
    # 1) Generate original (noise-free) image + ground truth
    # ---------------------------------------------------------
    width, height = 256, 256
    orig_img, square_coords, circle_coords = generate_synthetic_image(
        width=width,
        height=height,
        background_intensity=50,   # background
        square_intensity=200,      # square
        circle_intensity=180       # circle
    )
    gt_mask = generate_ground_truth_mask(width, height, square_coords, circle_coords)
    
    # ---------------------------------------------------------
    # 2) Apply edge detection with different thresholds
    # ---------------------------------------------------------
    canny_thresholds = [(50, 150), (100, 200)]
    sobel_thresholds = [50, 100, 150]
    laplacian_thresholds = [20, 50, 100]
    
    print("=== Original Image Evaluation ===")
    for (t1, t2) in canny_thresholds:
        edges_canny = edge_detection_canny(orig_img, t1, t2)
        p, r, f1 = evaluate_performance(edges_canny, gt_mask)
        print(f"Canny(t1={t1}, t2={t2}): Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        
    for th in sobel_thresholds:
        edges_sobel = edge_detection_sobel(orig_img, th)
        p, r, f1 = evaluate_performance(edges_sobel, gt_mask)
        print(f"Sobel(th={th}):         Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        
    for th in laplacian_thresholds:
        edges_lap = edge_detection_laplacian(orig_img, th)
        p, r, f1 = evaluate_performance(edges_lap, gt_mask)
        print(f"Laplacian(th={th}):   Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    
    # Example: display side by side for one set of thresholds (for visualization)
    # We'll pick one threshold setting each just as a demo:
    canny_demo = edge_detection_canny(orig_img, 50, 150)
    sobel_demo = edge_detection_sobel(orig_img, 100)
    lap_demo = edge_detection_laplacian(orig_img, 50)
    display_side_by_side("Original vs. Canny vs. Sobel vs. Laplacian (Demo)",
                         [orig_img, canny_demo, sobel_demo, lap_demo],
                         scale_factor=1.0)
    
    # ---------------------------------------------------------
    # 3) Add noise and evaluate again
    # ---------------------------------------------------------
    noisy_img = add_gaussian_noise(orig_img, sigma=20)
    print("\n=== Noisy Image Evaluation ===")
    for (t1, t2) in canny_thresholds:
        edges_canny = edge_detection_canny(noisy_img, t1, t2)
        p, r, f1 = evaluate_performance(edges_canny, gt_mask)
        print(f"Canny(t1={t1}, t2={t2}): Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        
    for th in sobel_thresholds:
        edges_sobel = edge_detection_sobel(noisy_img, th)
        p, r, f1 = evaluate_performance(edges_sobel, gt_mask)
        print(f"Sobel(th={th}):         Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        
    for th in laplacian_thresholds:
        edges_lap = edge_detection_laplacian(noisy_img, th)
        p, r, f1 = evaluate_performance(edges_lap, gt_mask)
        print(f"Laplacian(th={th}):   Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    
    # Display side by side for demo
    canny_demo_noisy = edge_detection_canny(noisy_img, 50, 150)
    sobel_demo_noisy = edge_detection_sobel(noisy_img, 100)
    lap_demo_noisy = edge_detection_laplacian(noisy_img, 50)
    display_side_by_side("Noisy vs. Canny vs. Sobel vs. Laplacian (Demo)",
                         [noisy_img, canny_demo_noisy, sobel_demo_noisy, lap_demo_noisy],
                         scale_factor=1.0)
    
    # ---------------------------------------------------------
    # 4) Vary intensities (e.g., higher background, lower object intensities)
    # ---------------------------------------------------------
    varied_bg = 80
    varied_square = 180
    varied_circle = 160
    varied_img, sq_coords_v, cir_coords_v = generate_synthetic_image(
        width=width,
        height=height,
        background_intensity=varied_bg,
        square_intensity=varied_square,
        circle_intensity=varied_circle
    )
    gt_mask_varied = generate_ground_truth_mask(width, height, sq_coords_v, cir_coords_v)
    
    print("\n=== Varied Intensity Image Evaluation ===")
    for (t1, t2) in canny_thresholds:
        edges_canny = edge_detection_canny(varied_img, t1, t2)
        p, r, f1 = evaluate_performance(edges_canny, gt_mask_varied)
        print(f"Canny(t1={t1}, t2={t2}): Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        
    for th in sobel_thresholds:
        edges_sobel = edge_detection_sobel(varied_img, th)
        p, r, f1 = evaluate_performance(edges_sobel, gt_mask_varied)
        print(f"Sobel(th={th}):         Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
        
    for th in laplacian_thresholds:
        edges_lap = edge_detection_laplacian(varied_img, th)
        p, r, f1 = evaluate_performance(edges_lap, gt_mask_varied)
        print(f"Laplacian(th={th}):   Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    
    # Display side by side for demo
    canny_demo_varied = edge_detection_canny(varied_img, 50, 150)
    sobel_demo_varied = edge_detection_sobel(varied_img, 100)
    lap_demo_varied = edge_detection_laplacian(varied_img, 50)
    display_side_by_side("Varied Intensity vs. Canny vs. Sobel vs. Laplacian (Demo)",
                         [varied_img, canny_demo_varied, sobel_demo_varied, lap_demo_varied],
                         scale_factor=1.0)
    
    # Wait for a key press to close images
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
