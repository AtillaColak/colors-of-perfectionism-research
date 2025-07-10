import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def visualize_step(image, title):
    """Helper function to display images during processing"""
    plt.figure(figsize=(10, 10))
    if len(image.shape) == 2:  # Grayscale
        plt.imshow(image, cmap='gray')
    else:  # RGB
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def detect_color_overflow_with_visualization(template_path: str, colored_path: str, threshold: float = 0.1):
    """
    Detect and visualize color overflow between regions in a colored mandala.
    """
    # loading the image
    template = cv2.imread(template_path)
    colored = cv2.imread(colored_path)
    
    # co vert to RGB 
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    # reize if there's a mismatch -> !this can be problematic maybe if the shape difference too much (also considering the base template is png and scans are jpg).
    if template.shape != colored.shape:
        colored = cv2.resize(colored, (template.shape[1], template.shape[0]))
    
    # template manipulation -> gaussian to reduce noise (otherwise I was getting too many contours overflow)
    template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    template_blur = cv2.GaussianBlur(template_gray, (5, 5), 0)
    # adaptive thresholding to create a binary mask of the template -- trying to higlight edges. I first tried using canny but different problems encountered. 
    template_binary = cv2.adaptiveThreshold(
        template_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # morph. operations to clean noise
    kernel = np.ones((3, 3), np.uint8)
    template_binary = cv2.morphologyEx(template_binary, cv2.MORPH_CLOSE, kernel)
    template_binary = cv2.morphologyEx(template_binary, cv2.MORPH_OPEN, kernel)
    
    # detecting contours. 
    contours, _ = cv2.findContours(template_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # did this to eliminate very small ones (uncolorably small ones). later I can include these but for now makes it easier to analyse. 
    min_area = template_binary.shape[0] * template_binary.shape[1] * 0.0001
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # initializing the overflow detection variables
    overflow_visualization = colored.copy()
    total_overflow_pixels = 0
    total_region_pixels = 0
    
    # loop through each region
    for contour in filtered_contours:
        # create region mask
        region_mask = np.zeros_like(template_binary)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        
        # dilating to include the neighboring pixels
        dilated_mask = cv2.dilate(region_mask, kernel, iterations=2)
        neighbor_mask = dilated_mask & ~region_mask
        
        # get region and neighbor colors
        region_pixels = colored[region_mask == 255]
        neighbor_pixels = colored[neighbor_mask == 255]
        
        if len(region_pixels) > 0:
            # mean color of the region
            mean_color = np.mean(region_pixels, axis=0)
            
            # overflow calculation by comparing neighbor pixels
            if len(neighbor_pixels) > 0:
                color_distances = np.linalg.norm(neighbor_pixels - mean_color, axis=1)
                overflow_points = color_distances < (threshold * 255)
                total_overflow_pixels += np.sum(overflow_points)
        
        # sum total region pixels
        total_region_pixels += np.sum(region_mask == 255)
    
    # visualize the overflow
    visualize_step(overflow_visualization, "Overflow Detection Visualization")
    
    # calculate overflow percentage
    overflow_percentage = (total_overflow_pixels / total_region_pixels) * 100 if total_region_pixels > 0 else 0
    
    return {
        "total_regions": len(filtered_contours),
        "overflow_pixels": total_overflow_pixels,
        "overflow_percentage": overflow_percentage
    }

if __name__ == "__main__":
    template_path = "mandala_with_apriltags.png"
    colored_path = "40033.jpg"
    
    try:
        # Detect overflow with visualization
        results = detect_color_overflow_with_visualization(template_path, colored_path)
        
        # Print final results
        print("\nFinal Results:")
        print(f"Total regions: {results['total_regions']}")
        print(f"Overflow pixels: {results['overflow_pixels']}")
        print(f"Overflow percentage: {results['overflow_percentage']:.2f}%")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")