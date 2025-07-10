import cv2
import numpy as np

def place_tag_on_corners(image, tag_image, position):
    """Place a single tag image at a specific corner of the given image."""
    h, w = image.shape[:2]
    th, tw = tag_image.shape[:2]

    if len(tag_image.shape) == 2:  
        tag_image = cv2.cvtColor(tag_image, cv2.COLOR_GRAY2BGR)

    if position == "top-left":
        image[0:th, 0:tw] = tag_image
    elif position == "top-right":
        image[0:th, w - tw:w] = tag_image
    elif position == "bottom-left":
        image[h - th:h, 0:tw] = tag_image
    elif position == "bottom-right":
        image[h - th:h, w - tw:w] = tag_image

def main():
    # Load the mandala image
    image_path = 'simple.jpg'  # Replace with your actual file path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")

    # Define PNG paths for each corner
    png_paths = {
        "top-left": 'tag_topleft.png',
        "top-right": 'tag_topright.png',
        "bottom-left": 'tag_bottomleft.png',
        "bottom-right": 'tag_bottomright.png'
    }

    # Load and place each tag image at the specified corner
    for position, png_path in png_paths.items():
        tag_image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if tag_image is None:
            raise ValueError(f"Tag image not found at {png_path}. Check file paths.")
        
        # Ensure the tag image is resized to the desired dimensions
        output_width, output_height = 100, 100  # Adjust as needed
        tag_image = cv2.resize(tag_image, (output_width, output_height))
        
        place_tag_on_corners(image, tag_image, position)

    # Save the output
    output_path = 'simple_with_apriltags.png'
    cv2.imwrite(output_path, image)
    print(f"Image saved with AprilTags at {output_path}")

if __name__ == "__main__":
    main()
