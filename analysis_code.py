import cv2
import numpy as np
import apriltag
from matplotlib import pyplot as plt 


def align_and_overlay(template_path, scan_path, output_path, alpha=0.5):
    # load the images
    template = cv2.imread(template_path)
    scan = cv2.imread(scan_path)

    # helper: preprocessing the images before apriltag detection
    def preprocess(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]

    template_proc = preprocess(template)
    scan_proc = preprocess(scan)

    # initialize apriltag detector
    options = apriltag.DetectorOptions(families='tag25h9')
    detector = apriltag.Detector(options)

    def detect_apriltags(color_image, binarized_image):
        detections = detector.detect(binarized_image)
        corners = []
        for detection in detections:
            # detection.corners is an ndarray of shape (4,2): [top-left, top-right, bottom-right, bottom-left]
            corners.append(detection.corners)
        return corners

    # detect tags in the template and scan
    template_tags = detect_apriltags(template, template_proc)
    scan_tags = detect_apriltags(scan, scan_proc)

    print("Number of AprilTags in template:", len(template_tags))
    print("Number of AprilTags in scan:", len(scan_tags))

    if len(template_tags) < 4 or len(scan_tags) < 4:
        print("Not enough AprilTags detected in one of the images.")
        return

    # For each image, we only use the first 4 tags 
    template_points = []
    scan_points = []
    for i in range(4):
        template_points.append(template_tags[i][0])  # corner index 0
        scan_points.append(scan_tags[i][0])          # corner index 0

    template_points = np.array(template_points, dtype="float32")
    scan_points = np.array(scan_points, dtype="float32")

    # compute initial homography via RANSAC
    homography_matrix, _ = cv2.findHomography(scan_points, template_points, cv2.RANSAC, 5.0)

    # warp the scan to align with the template (initial alignment)
    aligned_scan = cv2.warpPerspective(scan, homography_matrix,
                                       (template.shape[1], template.shape[0]))

    # ------------------------------------------------------------------------
    # ECC REFINEMENT STEP
    # ------------------------------------------------------------------------
    # Convert both images to grayscale
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_aligned = cv2.cvtColor(aligned_scan, cv2.COLOR_BGR2GRAY)

    # ECC can refine an initial homography. We start with the identity as a 3x3.
    warp_matrix_ecc = np.eye(3, 3, dtype=np.float32)

    # Set the ECC parameters
    number_of_iterations = 100
    termination_eps = 1e-6

    # findTransformECC expects both images to be 32-bit floats
    gray_template_f = gray_template.astype(np.float32)
    gray_aligned_f = gray_aligned.astype(np.float32)

    try:
        cc, warp_matrix_ecc = cv2.findTransformECC(
            gray_template_f,
            gray_aligned_f,
            warp_matrix_ecc,
            cv2.MOTION_HOMOGRAPHY,  # we want to refine a homography
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      number_of_iterations,
                      termination_eps)
        )

        # ECC’s warp matrix is typically applied in the INVERSE_MAP sense
        refined_aligned_scan = cv2.warpPerspective(
            aligned_scan,
            warp_matrix_ecc,
            (template.shape[1], template.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        print("ECC refinement successful. Correlation coefficient:", cc)

    except cv2.error as e:
        # If ECC fails, just stick with the original aligned_scan
        print("ECC refinement failed:", e)
        refined_aligned_scan = aligned_scan

    # ------------------------------------------------------------------------
    # Overlay the refined scan on the template
    # ------------------------------------------------------------------------
    overlay = cv2.addWeighted(template, alpha, refined_aligned_scan, 1 - alpha, 0)

    # save result
    cv2.imwrite(output_path, overlay)
    print(f"Aligned and overlaid image saved to {output_path}")

def create_processing_mask(height, width):
    """Create a mask that excludes the corner regions where AprilTags are located."""
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # cornersize calculating the april tag location.
    corner_size = min(height, width) // 20
    
    # mask out april tags.
    mask[:corner_size, :corner_size] = 0  # topleft
    mask[:corner_size, -corner_size:] = 0  # topright
    mask[-corner_size:, :corner_size] = 0  # bottomleft
    mask[-corner_size:, -corner_size:] = 0  # bottomright
    
    return mask

def load_and_preprocess(image_path, save_path="enhanced_binary.png"):
    """Load and preprocess the mandala image with enhanced borders."""
    # Step 1: Load the image in grayscale
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 2: Create a mask to exclude AprilTag regions
    processing_mask = create_processing_mask(original.shape[0], original.shape[1])
    
    # Step 3: Apply the mask to exclude AprilTag regions
    masked_img = cv2.bitwise_and(original, processing_mask)
    
    # Step 4: Apply thresholding to get the binary image
    _, binary = cv2.threshold(masked_img, 127, 255, cv2.THRESH_BINARY)
    
    # Step 5: Enhance borders using morphological operations
    kernel = np.ones((3, 3), np.uint8)  # Kernel size for morphological operations
    dilated = cv2.dilate(binary, kernel, iterations=1)  # Expand the borders
    enhanced_binary = cv2.erode(dilated, kernel, iterations=1)  # Sharpen the borders
    
    # Step 6: Save the enhanced binary output for debugging or visualization
    cv2.imwrite(save_path, enhanced_binary)
    print(f"Enhanced binary image saved to {save_path}")
    
    return enhanced_binary, original, processing_mask

def get_binary_versions(binary_img):
    """Create both versions of binary image (normal and inverted)."""
    _, binary = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    return binary, cv2.bitwise_not(binary)

def process_mandala_regions(image_path):
    """
    Process mandala regions by detecting contours from the enhanced binary image.
    Returns the filtered contours (assumed to be unique regions) and a regions map for visualization.
    """
    binary, original, processing_mask = load_and_preprocess(image_path)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Exclude the outer (largest) contour
    contours_to_process = contours[1:]
    contours_to_process = sorted(contours_to_process, key=cv2.contourArea)
    processed_mask = np.zeros_like(binary, dtype=np.uint8)
    filtered_contours = []
    regions_map = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    for contour in contours_to_process:
        contour_mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        unique_region_mask = cv2.subtract(contour_mask, processed_mask)
        if np.any(unique_region_mask):
            processed_mask = cv2.bitwise_or(processed_mask, unique_region_mask)
            area = cv2.contourArea(contour)
            if area > 50:  # ignore very small regions/noise
                filtered_contours.append(contour)
                random_color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.drawContours(regions_map, [contour], -1, random_color, thickness=-1)
    return filtered_contours, regions_map, binary

def analyze_color_region(region_hsv, region_mask, hue_tolerance=8,
                           BLACK_VALUE_THRESHOLD=5, BLACK_SATURATION_THRESHOLD=5,
                           WHITE_VALUE_THRESHOLD=180, WHITE_SATURATION_THRESHOLD=45,
                           MIN_COLOR_SATURATION=40, MIN_COLOR_VALUE=50):
    """
    For pixels within region_mask (nonzero), classify pixels as:
      - Black: if V < BLACK_VALUE_THRESHOLD and S < BLACK_SATURATION_THRESHOLD.
      - White: if V > WHITE_VALUE_THRESHOLD and S < WHITE_SATURATION_THRESHOLD.
      - Colored: if S >= MIN_COLOR_SATURATION and V >= MIN_COLOR_VALUE.
    
    Two modifications have been added:
      1) When computing the inner region (used for dominant hue), we exclude
         the outer ~2% of the region using a distance transform.
      2) Before doing overall hue classification, we pre‐classify border pixels
         (i.e. those in the outer 2%) that are not strictly black but whose S and V
         are very low (using lenient thresholds) as overflow.
    
    Then, among the remaining (non‐border) colored pixels, we compute the dominant hue 
    from the inner region and then classify each colored pixel as “correct” (if its hue is 
    within hue_tolerance of the dominant hue) or “overflow” (if it deviates more).
    
    Finally, if the overall painted (colored) area is less than ~8% of the region area,
    then the entire painted area is flagged as overflow.
    
    **Modification:** After classifying colored pixels, if the dominant hue covers less than 
    15% of the region (i.e. the “correct” pixels are less than 15% of the region area),
    then all colored pixels are marked as overflow.
    
    Returns a dictionary with counts and the two binary masks.
    """
    # Get all pixels from the region_mask (nonzero pixels)
    region_pixels = region_hsv[region_mask == 255]
    total_pixels = region_pixels.shape[0]

    # Initialize the output masks (same shape as region_mask)
    correct_mask = np.zeros_like(region_mask, dtype=np.uint8)
    overflow_mask = np.zeros_like(region_mask, dtype=np.uint8)

    if total_pixels == 0:
        return {"total": 0, "black": 0, "white": 0, "colored": 0,
                "correct": 0, "overflow": 0, "dominant_hue": None,
                "correct_mask": correct_mask, "overflow_mask": overflow_mask}

    # -------------------------------
    # 1. Count black and white pixels (using the strict thresholds)
    # -------------------------------
    black = np.sum((region_pixels[:, 2] < BLACK_VALUE_THRESHOLD) &
                   (region_pixels[:, 1] < BLACK_SATURATION_THRESHOLD))
    white = np.sum((region_pixels[:, 2] > WHITE_VALUE_THRESHOLD) &
                   (region_pixels[:, 1] < WHITE_SATURATION_THRESHOLD))

    # -------------------------------
    # 2. Compute inner and border masks via distance transform.
    #    (Here we remove the outer ~2% of the region.)
    # -------------------------------
    region_mask_bin = (region_mask == 255).astype(np.uint8)
    d = cv2.distanceTransform(region_mask_bin, cv2.DIST_L2, 5)
    max_d = d.max() if d.max() > 0 else 0
    inner_mask = np.zeros_like(region_mask)
    if max_d > 0:
        inner_mask[d > (0.05 * max_d)] = 255  # Using ~2% threshold.
    border_mask = cv2.subtract(region_mask, inner_mask)

    # -------------------------------
    # 3. Pre‐classify border pixels as overflow if they are "near‐black"
    #    (i.e. S and V are very low, though not strictly black).
    # -------------------------------
    BORDER_MARGIN = 45  # Adjust as needed.
    border_y, border_x = np.where(border_mask == 255)
    for i in range(len(border_y)):
        y, x = border_y[i], border_x[i]
        # region_hsv is assumed to have channels [H, S, V].
        h, s, v = region_hsv[y, x]
        if (s <= BLACK_SATURATION_THRESHOLD + BORDER_MARGIN) and (v <= BLACK_VALUE_THRESHOLD + BORDER_MARGIN):
            # If not strictly black, mark this border pixel as overflow.
            if not (v < BLACK_VALUE_THRESHOLD and s < BLACK_SATURATION_THRESHOLD):
                overflow_mask[y, x] = 255

    # -------------------------------
    # 4. Identify colored pixels (from the full region)
    # -------------------------------
    colored_bool = (region_pixels[:, 1] >= MIN_COLOR_SATURATION) & (region_pixels[:, 2] >= MIN_COLOR_VALUE)
    colored_pixels = region_pixels[colored_bool]
    count_colored = colored_pixels.shape[0]

    if count_colored == 0:
        dominant_hue = None
        correct = 0
        overflow = cv2.countNonZero(overflow_mask)
        return {"total": int(total_pixels),
                "black": int(black),
                "white": int(white),
                "colored": int(count_colored),
                "correct": int(correct),
                "overflow": int(overflow),
                "dominant_hue": dominant_hue,
                "correct_mask": correct_mask,
                "overflow_mask": overflow_mask}

    # -------------------------------
    # 5. Compute dominant hue from colored pixels in the inner region.
    #    If the inner region has no colored pixels, we assume the region is unpainted
    #    and mark all colored pixels as overflow.
    # -------------------------------
    inner_pixels = region_hsv[inner_mask == 255]
    if inner_pixels.size > 0:
        colored_bool_inner = (inner_pixels[:, 1] >= MIN_COLOR_SATURATION) & (inner_pixels[:, 2] >= MIN_COLOR_VALUE)
        colored_pixels_inner = inner_pixels[colored_bool_inner]
        if colored_pixels_inner.size == 0:
            # Fallback: inner region has no colored pixels.
            dominant_hue = None
            correct = 0
            ys_all, xs_all = np.where(region_mask == 255)
            region_hsv_pixels = region_hsv[ys_all, xs_all]
            is_colored_all = ((region_hsv_pixels[:, 1] >= MIN_COLOR_SATURATION) &
                              (region_hsv_pixels[:, 2] >= MIN_COLOR_VALUE))
            overflow_mask[ys_all[is_colored_all], xs_all[is_colored_all]] = 255
            overflow = int(np.sum(is_colored_all))
            return {"total": int(total_pixels),
                    "black": int(black),
                    "white": int(white),
                    "colored": int(count_colored),
                    "correct": int(correct),
                    "overflow": int(overflow),
                    "dominant_hue": dominant_hue,
                    "correct_mask": correct_mask,
                    "overflow_mask": overflow_mask}
        else:
            hues = colored_pixels_inner[:, 0].astype(np.uint8)
            hist = np.bincount(hues, minlength=181)
            dominant_hue = int(np.argmax(hist))
    else:
        # If inner_mask is completely empty, mark all colored pixels as overflow.
        dominant_hue = None
        correct = 0
        ys_all, xs_all = np.where(region_mask == 255)
        region_hsv_pixels = region_hsv[ys_all, xs_all]
        is_colored_all = ((region_hsv_pixels[:, 1] >= MIN_COLOR_SATURATION) &
                          (region_hsv_pixels[:, 2] >= MIN_COLOR_VALUE))
        overflow_mask[ys_all[is_colored_all], xs_all[is_colored_all]] = 255
        overflow = int(np.sum(is_colored_all))
        return {"total": int(total_pixels),
                "black": int(black),
                "white": int(white),
                "colored": int(count_colored),
                "correct": int(correct),
                "overflow": int(overflow),
                "dominant_hue": dominant_hue,
                "correct_mask": correct_mask,
                "overflow_mask": overflow_mask}

    # -------------------------------
    # 6. Classify all colored pixels in the full region based on their hue difference.
    # -------------------------------
    ys, xs = np.where(region_mask == 255)
    region_hsv_pixels = region_hsv[ys, xs]  # shape (N, 3)
    is_colored = ((region_hsv_pixels[:, 1] >= MIN_COLOR_SATURATION) &
                  (region_hsv_pixels[:, 2] >= MIN_COLOR_VALUE))
    pixel_hues = region_hsv_pixels[:, 0].astype(np.int32)
    # Compute circular hue difference:
    diff = np.minimum(np.abs(pixel_hues - dominant_hue),
                      180 - np.abs(pixel_hues - dominant_hue))
    is_correct = (diff <= hue_tolerance) & is_colored
    is_overflow = (diff > hue_tolerance) & is_colored

    correct = int(np.sum(is_correct))
    overflow = int(np.sum(is_overflow))

    correct_mask[ys[is_correct], xs[is_correct]] = 255
    overflow_mask[ys[is_overflow], xs[is_overflow]] = 255

    # -------------------------------
    # NEW CONDITION: Check if the dominant hue covers at least 15% of the region.
    # If not, mark all colored pixels as overflow.
    # -------------------------------
    if total_pixels > 0 and correct < 0.15 * total_pixels:
        correct = 0
        # Clear the correct_mask since we are not accepting any as "correct"
        correct_mask[:] = 0
        # Mark all colored pixels (in the region) as overflow.
        ys_all, xs_all = np.where(region_mask == 255)
        region_hsv_pixels_all = region_hsv[ys_all, xs_all]
        is_colored_all = ((region_hsv_pixels_all[:, 1] >= MIN_COLOR_SATURATION) &
                          (region_hsv_pixels_all[:, 2] >= MIN_COLOR_VALUE))
        overflow_mask = np.zeros_like(region_mask, dtype=np.uint8)
        overflow_mask[ys_all[is_colored_all], xs_all[is_colored_all]] = 255
        overflow = int(np.sum(is_colored_all))

    # -------------------------------
    # 7. (Optional) If the painted area is very small (< 8% of the region),
    #    then treat all colored pixels as overflow.
    # -------------------------------
    if total_pixels > 0 and (count_colored / total_pixels) < 0.08:
        correct = 0
        correct_mask[:] = 0
        overflow_mask = np.zeros_like(region_mask, dtype=np.uint8)
        ys_all, xs_all = np.where(region_mask == 255)
        all_pixels = region_hsv[ys_all, xs_all]
        is_colored_all = ((all_pixels[:, 1] >= MIN_COLOR_SATURATION) &
                          (all_pixels[:, 2] >= MIN_COLOR_VALUE))
        overflow_mask[ys_all[is_colored_all], xs_all[is_colored_all]] = 255
        overflow = int(np.sum(is_colored_all))

    return {"total": int(total_pixels),
            "black": int(black),
            "white": int(white),
            "colored": int(count_colored),
            "correct": int(correct),
            "overflow": int(overflow),
            "dominant_hue": dominant_hue,
            "correct_mask": correct_mask,
            "overflow_mask": overflow_mask}

def analyze_regions(template_path, aligned_scan_path, output_visualization_path, debug=True):
    """
    Analyzes mandala regions defined by contours in the template image.
    For each unique region, the aligned scan (in HSV) is used to compute:
      - The dominant hue among colored pixels (computed only from the inner region).
      - A classification into correct (within hue tolerance) vs. overflow (exceeding tolerance).
    A debug visualization image is created where correctly colored pixels are overlaid in green
    and overflow pixels in red.
    
    This version uses a global mask (processed_mask) so that once a pixel is assigned
    to a region, it is excluded from later (usually larger) regions.
    """
    # Get regions (contours, regions_map, and a binary image) from the template
    filtered_contours, regions_map, binary = process_mandala_regions(template_path)
    
    # Load the aligned scan and apply cleaning before converting to HSV
    aligned_scan = cv2.imread(aligned_scan_path)
    aligned_scan = cv2.medianBlur(aligned_scan, 5)
    kernel_small = np.ones((2, 2), np.uint8)
    closed_small = cv2.morphologyEx(aligned_scan, cv2.MORPH_CLOSE, kernel_small)
    kernel_medium = np.ones((3, 3), np.uint8)
    cleaned_scan = cv2.morphologyEx(closed_small, cv2.MORPH_OPEN, kernel_medium)
    aligned_scan_hsv = cv2.cvtColor(cleaned_scan, cv2.COLOR_BGR2HSV)

    # Create a debug visualization image (copy of the cleaned scan)
    debug_vis = cleaned_scan.copy()

    # Threshold parameters (tweaked for pastel colors and stricter colored pixel identification)
    BLACK_VALUE_THRESHOLD = 15
    BLACK_SATURATION_THRESHOLD = 15
    WHITE_VALUE_THRESHOLD = 170
    WHITE_SATURATION_THRESHOLD = 48
    MIN_COLOR_SATURATION = 48
    MIN_COLOR_VALUE = 45
    hue_tolerance = 10

    # Global accumulators for overall statistics
    total_pixels = 0
    total_black = 0
    total_white = 0
    total_colored = 0
    total_correct = 0
    total_overflow = 0

    # Create a mask to track which pixels have already been processed.
    # This ensures that if a smaller (inner) region is processed first,
    # its pixels will be excluded from later (outer) regions.
    processed_mask = np.zeros_like(binary, dtype=np.uint8)

    for idx, contour in enumerate(filtered_contours):
        # Create a mask for the current contour (all pixels inside this region)
        contour_mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        
        # Remove any pixels that have already been assigned to a previous region.
        unique_mask = cv2.subtract(contour_mask, processed_mask)
        if not np.any(unique_mask):
            continue  # Skip this region if no unique pixels remain.
        
        # Update the processed_mask so that these pixels are not counted again.
        processed_mask = cv2.bitwise_or(processed_mask, unique_mask)
        
        # Analyze the unique region using the helper function.
        stats = analyze_color_region(aligned_scan_hsv, unique_mask,
                                     hue_tolerance=hue_tolerance,
                                     BLACK_VALUE_THRESHOLD=BLACK_VALUE_THRESHOLD,
                                     BLACK_SATURATION_THRESHOLD=BLACK_SATURATION_THRESHOLD,
                                     WHITE_VALUE_THRESHOLD=WHITE_VALUE_THRESHOLD,
                                     WHITE_SATURATION_THRESHOLD=WHITE_SATURATION_THRESHOLD,
                                     MIN_COLOR_SATURATION=MIN_COLOR_SATURATION,
                                     MIN_COLOR_VALUE=MIN_COLOR_VALUE)
        
        # Accumulate statistics from this region.
        total_pixels += stats["total"]
        total_black += stats["black"]
        total_white += stats["white"]
        total_colored += stats["colored"]
        total_correct += stats["correct"]
        total_overflow += stats["overflow"]

        # Overlay the classification results on the debug image:
        ys, xs = np.where(stats["correct_mask"] == 255)
        debug_vis[ys, xs] = [0, 255, 0]  # Green for correctly colored pixels.
        ys, xs = np.where(stats["overflow_mask"] == 255)
        debug_vis[ys, xs] = [0, 0, 255]  # Red for overflow pixels.

        # Draw the contour and label (for visual reference).
        cv2.drawContours(debug_vis, [contour], -1, (255, 255, 0), 1)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv2.putText(debug_vis, f"R{idx}", (cX - 20, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        region_pct = (stats["colored"] + stats["overflow"]) / stats["total"] * 100 if stats["total"] else 0
        overflow_pct = (stats["overflow"] / stats["colored"] * 100) if stats["colored"] else 0
        if debug:
            print(f"Region {idx}: Total={stats['total']} pixels, "
                  f"Black={stats['black']}, White={stats['white']}, "
                  f"Colored={stats['colored']} (Correct: {stats['correct']}, Overflow: {stats['overflow']}), "
                  f"Dominant Hue={stats['dominant_hue']}, "
                  f"Painted Area={region_pct:.1f}%, Overflow Ratio={overflow_pct:.1f}%")

    overall_coverage_pct = (total_correct / total_pixels * 100) if total_pixels else 0
    overall_overflow_pct = (total_overflow / total_pixels * 100) if total_pixels else 0

    if debug:
        print("\nOverall Mandala Analysis:")
        print(f"  Total Mandala Area (pixels): {total_pixels}")
        print(f"  Total Black Pixels: {total_black}")
        print(f"  Total White Pixels: {total_white}")
        print(f"  Total Colored Pixels: {total_colored}")
        print(f"    - Correctly colored: {total_correct}")
        print(f"    - Overflow: {total_overflow}")
        print(f"  Overall Coverage (Correct Color) Percentage: {overall_coverage_pct:.2f}%")
        print(f"  Overall Overflow Percentage: {overall_overflow_pct:.2f}%")

    cv2.imwrite(output_visualization_path, debug_vis)
    print(f"Visualization saved to {output_visualization_path}")

def main():
    template_path = "mandala_with_apriltags.png"
    scan_path = "40110.jpg"
    aligned_output_path = "aligned_scan.png"
    visualization_output_path = f"{scan_path[0:5]}_debug.png"
    
    # unblended version for analysis
    align_and_overlay(template_path, scan_path, "aligned_scan_analysis.png", alpha=0.0)
    
    # Create a separate blended version for display/debug
    align_and_overlay(template_path, scan_path, aligned_output_path, alpha=0.7)
    
    # analyze regions using the unblended aligned scan
    analyze_regions(template_path, "aligned_scan_analysis.png", visualization_output_path, debug=True)
    
if __name__ == "__main__":
    main()


# TODO: 
# 1) It should detect more colors better saturation. 
# 2) noise fixing -> sometimes there are few pixels identiified as colored/overflow in random parts of the mandala even thouhg there is no color near those pixels at all. so these are clearly error and should be ignored in the result.
# 3) I'll IGNORE FOR NOW: maybe later -> include the checking of black circles thing. 