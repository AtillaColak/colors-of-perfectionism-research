import cv2
import apriltag
import numpy as np

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

# paths
template_path = "mandala_with_apriltags.png"
scan_path = "40000.jpg"
output_path = "aligned_overlay.png"

def add_text(image, text, position, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image

# loading the images
template = cv2.imread(template_path)
scan = cv2.imread(scan_path)

# adding debug text to the images -- both should be visible.
template = add_text(template, "Template Text", (50, 50), (0, 255, 0))  # Green text on template
scan = add_text(scan, "Scan Text", (200, 200), (255, 0, 0))  # Blue text on scan

# saving the modified images (optional for debugging)
cv2.imwrite("template_with_text.png", template)
cv2.imwrite("scan_with_text.png", scan)

# call with the text-inserted scans. 
align_and_overlay("template_with_text.png", "scan_with_text.png", "aligned_overlay_with_text.png")
