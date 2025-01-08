import cv2
import numpy as np
import apriltag

def align_and_overlay(template_path, scan_path, output_path, alpha=0.5):
    # load the images
    template = cv2.imread(template_path)
    scan = cv2.imread(scan_path)

    # preprocessing the images before apriltag detection to reduce noise and potential bugs/inaccuracies. 
    def preprocess(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]    
    
    template_proc = preprocess(template)
    scan_proc = preprocess(scan)

    # initializing apriltag detector with the proper family 
    options = apriltag.DetectorOptions(families='tag25h9')
    detector = apriltag.Detector(options)

    def detect_apriltags(image, processed):
        detections = detector.detect(processed)
        corners = []
        for detection in detections:
            corners.append(detection.corners)
        return corners

    # detect tags in the template and scan
    template_tags = detect_apriltags(template, template_proc)
    scan_tags = detect_apriltags(scan, scan_proc)

    print(len(template_tags))
    print(len(scan_tags))

    if len(template_tags) < 4 or len(scan_tags) < 4:
        print("Not enough AprilTags detected in one of the images.")
        return

    # only 4 detected tags for alignment (assuming the same order)
    template_points = np.array([corner[0] for corner in template_tags[:4]], dtype="float32")
    scan_points = np.array([corner[0] for corner in scan_tags[:4]], dtype="float32")

    # computing homography
    homography_matrix, _ = cv2.findHomography(scan_points, template_points)

    # warping the scan to align with the template
    aligned_scan = cv2.warpPerspective(scan, homography_matrix, (template.shape[1], template.shape[0]))

    # Overlay the aligned scan on the template
    overlay = cv2.addWeighted(template, alpha, aligned_scan, 1-alpha, 0)

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

def load_and_preprocess(image_path):
    """Load and preprocess the mandala image."""
    # reading image in grayscale
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Add denoising
    denoised = cv2.fastNlMeansDenoising(original)
    
    processing_mask = create_processing_mask(denoised.shape[0], denoised.shape[1])
    masked_img = cv2.bitwise_and(denoised, processing_mask)
    
    return cv2.threshold(masked_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1], original, processing_mask


def get_binary_versions(binary_img):
    """Create both versions of binary image (normal and inverted)."""
    _, binary = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    return binary, cv2.bitwise_not(binary)


def process_mandala(image_path):
    """Process the mandala image and return contours and region information."""
    # load and preprocess image
    binary, original, processing_mask = load_and_preprocess(image_path)
    
    # find contours in the masked binary image
    contours, hierarchy = cv2.findContours(binary, 
                                         cv2.RETR_TREE, 
                                         cv2.CHAIN_APPROX_SIMPLE)
    
    # Create regions map for analysis
    regions_map = np.zeros_like(binary[..., None]).repeat(3, axis=2)    
    return contours, regions_map, processing_mask
    
def analyze_regions(template_path, aligned_scan_path, debug=True):
    # Load and preprocess the template to get regions
    contours, regions_map, _ = process_mandala(template_path)

    # Load aligned scan
    aligned_scan = cv2.imread(aligned_scan_path)
    aligned_scan_hsv = cv2.cvtColor(aligned_scan, cv2.COLOR_BGR2HSV)

    region_analysis = {}

    # Parameters for color detection
    BLACK_VALUE_THRESHOLD = 20  # Stricter maximum value for black detection
    BLACK_SATURATION_THRESHOLD = 20  # Stricter maximum saturation for black detection
    WHITE_VALUE_THRESHOLD = 240  # Minimum value for white detection
    WHITE_SATURATION_THRESHOLD = 30  # Maximum saturation for white detection
    COLOR_HUE_TOLERANCE = 8  # Stricter hue variation tolerance for overflow
    MIN_COLOR_SATURATION = 50  # Higher minimum saturation for colored pixels
    MIN_COLOR_VALUE = 70  # Higher minimum value for colored pixels

    for idx, contour in enumerate(contours):
        # Create a mask for the current region
        mask = np.zeros_like(regions_map[..., 0])
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)  # Fill the contour

        # Extract the region from the aligned scan using the mask
        region_pixels = cv2.bitwise_and(aligned_scan, aligned_scan, mask=mask)

        # Calculate region statistics
        region_hsv_values = aligned_scan_hsv[mask == 255]
        total_pixels = np.sum(mask == 255)

        # Ignore black lines or similar tones (stricter low V and S values)
        black_line_mask = (region_hsv_values[..., 1] <= BLACK_SATURATION_THRESHOLD) & \
                          (region_hsv_values[..., 2] <= BLACK_VALUE_THRESHOLD)
        black_pixels = np.sum(black_line_mask)

        # Identify white pixels
        white_mask = (region_hsv_values[..., 1] <= WHITE_SATURATION_THRESHOLD) & \
                     (region_hsv_values[..., 2] >= WHITE_VALUE_THRESHOLD)
        white_pixels = np.sum(white_mask)

        # Identify colored pixels
        color_mask = (region_hsv_values[..., 1] >= MIN_COLOR_SATURATION) & \
                     (region_hsv_values[..., 2] >= MIN_COLOR_VALUE)
        colored_pixels = np.sum(color_mask)

        # Find the most common color in the region
        dominant_hue = None
        overflow_pixels = 0

        if colored_pixels > 0:
            colored_hues = region_hsv_values[color_mask, 0]

            # Create histogram of hues
            hist, bins = np.histogram(colored_hues, bins=36, range=(0, 180))
            dominant_hue_bin = np.argmax(hist)
            dominant_hue = (bins[dominant_hue_bin] + bins[dominant_hue_bin + 1]) / 2

            # Check for overflow - pixels with significantly different hues
            hue_diff = np.minimum(
                np.abs(colored_hues - dominant_hue),
                180 - np.abs(colored_hues - dominant_hue)
            )
            overflow_mask = hue_diff > COLOR_HUE_TOLERANCE
            overflow_pixels = np.sum(overflow_mask)

            # Adjust colored pixels count to exclude overflow
            colored_pixels -= overflow_pixels

        region_analysis[idx] = {
            "total_pixels": total_pixels,
            "colored_pixels": int(colored_pixels),
            "white_pixels": int(white_pixels),
            "black_pixels": int(black_pixels),
            "overflow_pixels": int(overflow_pixels),
            "dominant_hue": dominant_hue
        }

        if debug:
            print(f"\nRegion {idx}:")
            print(f"Total Pixels: {total_pixels}")
            print(f"Colored Pixels: {colored_pixels} ({(colored_pixels/total_pixels)*100:.1f}%)")
            print(f"White Pixels: {white_pixels} ({(white_pixels/total_pixels)*100:.1f}%)")
            print(f"Black Pixels: {black_pixels} ({(black_pixels/total_pixels)*100:.1f}%)")
            print(f"Overflow Pixels: {overflow_pixels} ({(overflow_pixels/total_pixels)*100:.1f}%)")
            if dominant_hue is not None:
                print(f"Dominant Hue: {dominant_hue:.1f}")

    return region_analysis

def main():
    template_path = "mandala_with_apriltags.png"
    scan_path = "40000.jpg"
    aligned_output_path = "aligned_scan.png"
    
    # unblended version for analysis
    align_and_overlay(template_path, scan_path, "aligned_scan_analysis.png", alpha=0.0)
    
    # Create a separate blended version for display/debug
    align_and_overlay(template_path, scan_path, aligned_output_path, alpha=0.7)
    
    # analyze regions using the unblended aligned scan
    region_analysis = analyze_regions(template_path, "aligned_scan_analysis.png", debug=True)
    
    # calculate overall statistics
    total_mandala_pixels = 0
    total_colored_pixels = 0
    total_white_pixels = 0
    total_overflow_pixels = 0
    
    for region_id, analysis in region_analysis.items():
        total_mandala_pixels += analysis['total_pixels']
        total_colored_pixels += analysis['colored_pixels']
        total_white_pixels += analysis['white_pixels']
        total_overflow_pixels += analysis['overflow_pixels']
    
    # printing overall analysis
    print("\nOverall Mandala Analysis:")
    print(f"Total number of regions: {len(region_analysis)}")
    print(f"Total Mandala Area (pixels): {total_mandala_pixels}")
    print(f"Total Colored Pixels: {total_colored_pixels}")
    print(f"Total White Pixels: {total_white_pixels}")
    print(f"Total Overflow Pixels: {total_overflow_pixels}")
    print(f"Overall Coverage Percentage: {(total_colored_pixels/total_mandala_pixels)*100:.2f}%")
    print(f"Overall Overflow Percentage: {(total_overflow_pixels/total_mandala_pixels)*100:.2f}%")
    print(f"Total Coloring Percentage (including overflow): {((total_colored_pixels + total_overflow_pixels)/total_mandala_pixels)*100:.2f}%")

if __name__ == "__main__":
    main()
