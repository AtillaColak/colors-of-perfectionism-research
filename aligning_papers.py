# TODO: the aligning is not proper because resizing the example changes the pixels and locations of markers across both images. Find_homography? 
# TODO: fix alignment.
import cv2
import numpy as np
import apriltag

def detect_markers_Aruco(image, border_color):
    """Detect AprilTags using OpenCV's ArUco module."""
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the dictionary for AprilTags
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers in the image
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    marker_positions = []
    if ids is not None:
        for i, corner in zip(ids.flatten(), corners):
            cv2.polylines(image, [corner[0].astype(int)], isClosed=True, color=border_color, thickness=2)

            # Get bounding box coordinates
            x_min = int(np.min(corner[0][:, 0]))
            y_min = int(np.min(corner[0][:, 1]))
            x_max = int(np.max(corner[0][:, 0]))
            y_max = int(np.max(corner[0][:, 1]))
            marker_positions.append((x_min, y_min, x_max - x_min, y_max - y_min))

    print(f"AprilTags detected: {len(marker_positions)}")
    return marker_positions


def detect_markers(image, border_color):
    """Detect AprilTags in the image and return their positions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the AprilTag detector with the Tag 25H9 family
    options = apriltag.DetectorOptions(families='tag25h9', debug=0)  # Set debug to 0
    detector = apriltag.Detector(options)

    # Detect tags in the entire image
    results = detector.detect(gray)

    marker_positions = []
    for r in results:
        corners = r.corners.astype(int)

        # Skip small or low-confidence detections
        if len(corners) < 4:  # Ensure we have at least 4 corners
            continue

        # Draw the detected marker with the specified border color
        cv2.polylines(image, [corners], isClosed=True, color=border_color, thickness=2)

        # Store the bounding box as (x, y, width, height)
        x_min = int(np.min(corners[:, 0]))
        y_min = int(np.min(corners[:, 1]))
        x_max = int(np.max(corners[:, 0]))
        y_max = int(np.max(corners[:, 1]))
        marker_positions.append((x_min, y_min, x_max - x_min, y_max - y_min))

    print(f"AprilTags detected: {len(marker_positions)}")  # Print number of markers found
    return marker_positions

import cv2
import numpy as np

def align_images(base_image, overlay_image, marker_positions_base, marker_positions_overlay):
    """Align the overlay image with the base image using translation, rotation, and scaling."""

    # Check if there are enough markers in both images
    if len(marker_positions_base) < 2 or len(marker_positions_overlay) < 2:
        raise ValueError("At least two markers are required in each image for alignment.")
    
    # Calculate centers of the detected markers
    base_centers = np.array([(x + w // 2, y + h // 2) for (x, y, w, h) in marker_positions_base], dtype=np.float32)
    overlay_centers = np.array([(x + w // 2, y + h // 2) for (x, y, w, h) in marker_positions_overlay], dtype=np.float32)
    
    # Calculate translation by aligning the centers of the two sets of points
    translation = np.mean(base_centers, axis=0) - np.mean(overlay_centers, axis=0)

    # Apply translation to overlay centers
    overlay_centers_translated = overlay_centers + translation
    
    # Calculate scaling factor using the average distance between markers
    # in both images to get a more precise relative scale.
    base_distances = [np.linalg.norm(base_centers[i] - base_centers[j])
                      for i in range(len(base_centers)) for j in range(i + 1, len(base_centers))]
    overlay_distances = [np.linalg.norm(overlay_centers[i] - overlay_centers[j])
                         for i in range(len(overlay_centers)) for j in range(i + 1, len(overlay_centers))]

    # Use the ratio of average distances as the scaling factor
    scale = np.mean(base_distances) / np.mean(overlay_distances)
    
    # Calculate rotation angle by finding the angle between vectors formed by markers
    angle_base = np.arctan2((base_centers[1][1] - base_centers[0][1]), (base_centers[1][0] - base_centers[0][0]))
    angle_overlay = np.arctan2((overlay_centers_translated[1][1] - overlay_centers_translated[0][1]), 
                               (overlay_centers_translated[1][0] - overlay_centers_translated[0][0]))
    rotation_angle = np.degrees(angle_base - angle_overlay)
    
    # Define rotation matrix
    rows, cols, _ = overlay_image.shape
    center = (cols / 2, rows / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    
    # Apply rotation and scaling to overlay image
    aligned_overlay = cv2.warpAffine(overlay_image, rotation_matrix, (cols, rows))
    
    # Translate the aligned overlay image
    M = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    aligned_overlay = cv2.warpAffine(aligned_overlay, M, (cols, rows))
    
    # Combine images to visualize alignment
    combined_image = cv2.addWeighted(base_image, 0.5, aligned_overlay, 0.5, 0)

    return combined_image

def add_text(image, text, position):
    """Add text to the image at a specified position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, 1, (0, 0, 0), 2, cv2.LINE_AA)

def main():
    # Load the base mandala image
    base_image_path = 'mandala_with_apriltags.png'
    base_image = cv2.imread(base_image_path)
    if base_image is None:
        raise ValueError("Base image not found or path is incorrect.")

    # Load the printed and scanned mandala image
    overlay_image_path = 'mandala_example.png'
    overlay_image = cv2.imread(overlay_image_path)
    if overlay_image is None:
        raise ValueError("Overlay image not found or path is incorrect.")
    
    overlay_image = cv2.resize(overlay_image, (base_image.shape[1], base_image.shape[0]))

    # Detect markers in both images with specified colors
    marker_positions_base = detect_markers_Aruco(base_image.copy(), (0, 0, 255))  # Red for base image
    marker_positions_overlay = detect_markers_Aruco(overlay_image.copy(), (0, 255, 0))  # Green for overlay image

    # add_text(base_image, "base", (50,50))

    if marker_positions_base and marker_positions_overlay:
        # Align the overlay image with the base image
        aligned_overlay = align_images(base_image, overlay_image, marker_positions_base, marker_positions_overlay)

        # Save the aligned image
        output_path = 'aligned_mandala.png'
        cv2.imwrite(output_path, aligned_overlay)
        print(f"Aligned image saved at {output_path}")

    else:
        print("No markers detected in one or both images.")

if __name__ == "__main__":
    main()
