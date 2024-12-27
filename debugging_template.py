import cv2
import numpy as np
import random

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
    
    # masking to exclude AprilTag regions
    processing_mask = create_processing_mask(original.shape[0], original.shape[1])
    
    # excluding apriltags version of the image. 
    masked_img = cv2.bitwise_and(original, processing_mask)
    
    # applying threshold to get binary image for processing
    _, binary = cv2.threshold(masked_img, 127, 255, cv2.THRESH_BINARY)
    
    return binary, original, processing_mask

def get_binary_versions(binary_img):
    """Create both versions of binary image (normal and inverted)."""
    _, binary = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    return binary, cv2.bitwise_not(binary)

def create_random_colored_regions(binary_img, original_img, processing_mask):
    """Fill regions with random colors while preserving apriltags."""
    # create a 3-channel image for colored output
    colored = np.zeros_like(binary_img[..., None]).repeat(3, axis=2)
    
    # find contours in the masked binary image
    contours, hierarchy = cv2.findContours(binary_img, 
                                         cv2.RETR_TREE, 
                                         cv2.CHAIN_APPROX_SIMPLE)
    
    # filling each region with a random color
    for contour in contours:
        color = (random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255))
        cv2.fillPoly(colored, [contour], color)
    
    # keeping only the colored regions within the processing mask
    colored = cv2.bitwise_and(colored, colored, mask=processing_mask)
    
    # create inverted processing mask for apriltag
    apriltag_mask = cv2.bitwise_not(processing_mask)
    
    # adding original apriltags back
    apriltag_regions = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    apriltag_regions = cv2.bitwise_and(apriltag_regions, apriltag_regions, mask=apriltag_mask)
    
    # combine colored regions with original apriltags
    result = cv2.add(colored, apriltag_regions)
    
    return result

def create_sequential_colored_regions(binary_img, original_img, processing_mask):
    """Create image with sequentially numbered regions while preserving AprilTags."""
    # create a 3-channel image for colored output
    sequential = np.zeros_like(binary_img[..., None]).repeat(3, axis=2)
    
    # find contours in the masked binary image
    contours, hierarchy = cv2.findContours(binary_img, 
                                         cv2.RETR_TREE, 
                                         cv2.CHAIN_APPROX_SIMPLE)
    
    # counter for regions
    region_count = 1
    
    # filling each region sequentially
    for contour in contours:
        r = region_count 
        g = 0 
        b = 0 
        if r > 255:
            g = r - 255
            r = 0
            if g > 255:
                b = g - 255
                g = 0 
        color = (b, g, r)
        cv2.fillPoly(sequential, [contour], color)
        region_count += 1
    
    # keep only the colored regions within the processing mask
    sequential = cv2.bitwise_and(sequential, sequential, mask=processing_mask)
    
    # same apriltag addition logic as above.
    apriltag_mask = cv2.bitwise_not(processing_mask)
    
    apriltag_regions = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    apriltag_regions = cv2.bitwise_and(apriltag_regions, apriltag_regions, mask=apriltag_mask)
    
    result = cv2.add(sequential, apriltag_regions)
    
    return result

def process_mandala(image_path):
    """Process the mandala image and return all requested versions."""
    # load and preprocess image
    binary, original, processing_mask = load_and_preprocess(image_path)
    
    # get both binary versions
    binary_normal, binary_inverted = get_binary_versions(original)
    
    # create random colored version
    random_colored = create_random_colored_regions(binary, original, processing_mask)
    
    # create sequential colored version
    sequential_colored = create_sequential_colored_regions(binary, original, processing_mask)
    
    return {
        'binary_normal': binary_normal,
        'binary_inverted': binary_inverted,
        'random_colored': random_colored,
        'sequential_colored': sequential_colored
    }

def save_results(results, output_prefix):
    """Save all processed versions of the image."""
    for name, img in results.items():
        output_path = f"{output_prefix}_{name}.png"
        cv2.imwrite(output_path, img)
        
if __name__ == "__main__":
    input_path = "mandala_with_apriltags.png" 
    output_prefix = "debugging_template_"
    
    results = process_mandala(input_path)
    save_results(results, output_prefix)