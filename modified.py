import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Preprocessing function to resize, grayscale, and normalize images
# def preprocess_image(image_path):
#     # Load image
#     image = cv2.imread(image_path)
#     # Convert to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Resize to standard size (512x512)
#     resized_image = cv2.resize(gray_image, (512, 512))
#     # Normalize
#     normalized_image = resized_image / 255.0
#     return normalized_image, resized_image

def preprocess_image(image_path):
  
  try:
    # Load image
    image = cv2.imread(image_path)
    if image is None:
      raise ValueError(f"Error reading image: {image_path}")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to standard size (512x512)
    resized_image = cv2.resize(gray_image, (512, 512))

    # Normalize
    normalized_image = resized_image / 255.0

    return normalized_image, resized_image
  except Exception as e:
    print(f"Error preprocessing image: {e}")
    return None  # Or handle the error differently

# Function to compute edge detection (helps identify road surface changes)
def edge_detection(image):
    edges = cv2.Canny((image * 255).astype(np.uint8), threshold1=100, threshold2=200)
    return edges

# Function to detect utility ducts via line detection (using Hough Transform)
def detect_utility_ducts(image):
    edges = edge_detection(image)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=5)
    if lines is not None:
        return len(lines)  # Number of detected lines
    return 0  # No lines detected

# Function to compute similarity score using Structural Similarity Index (SSIM)
def compute_similarity(image1, image2):
    similarity, _ = ssim(image1, image2, full=True, data_range=1.0)
    return similarity

# Function to calculate pixel intensity difference between two images
def pixel_intensity_difference(image1, image2):
    return np.mean(np.abs(image1 - image2))

# Function to calculate road coverage percentage based on edge detection
def road_coverage_change(edge_image1, edge_image2):
    road_pixels1 = np.sum(edge_image1 > 0)
    road_pixels2 = np.sum(edge_image2 > 0)
    if road_pixels1 == 0:  # Avoid division by zero
        return 0
    return (road_pixels2 - road_pixels1) / road_pixels1 * 100

# Function to calculate histogram difference (based on intensity levels)
def histogram_difference(image1, image2):
    image1_uint8 = (image1 * 255).astype(np.uint8)
    image2_uint8 = (image2 * 255).astype(np.uint8)
    hist1 = cv2.calcHist([image1_uint8], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2_uint8], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Function to detect keypoints for macadamization and pedestrian infrastructure changes
def keypoint_detection(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

# Main function to process two images and calculate construction progress
def calculate_construction_progress(image_path1, image_path2):
    # Preprocess both images
    image1, resized_image1 = preprocess_image(image_path1)
    image2, resized_image2 = preprocess_image(image_path2)

    # Detect edges in both images
    edges_image1 = edge_detection(image1)
    edges_image2 = edge_detection(image2)

    # Calculate SSIM (Structural Similarity Index)
    similarity_score = compute_similarity(image1, image2)

    # Calculate road coverage change based on edges
    road_coverage_change_percent = road_coverage_change(edges_image1, edges_image2)

    # Calculate pixel intensity difference
    pixel_diff = pixel_intensity_difference(image1, image2)

    # Calculate histogram difference (intensity correlation)
    hist_diff = histogram_difference(image1, image2)

    # ORB keypoint detection for macadamization and pedestrian infrastructure
    kp1, kp2, matches = keypoint_detection(resized_image1, resized_image2)

    # Detect utility ducts
    utility_ducts_change = detect_utility_ducts(image1) - detect_utility_ducts(image2)

    # Draw keypoint matches
    img_matches = cv2.drawMatches(resized_image1, kp1, resized_image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Combining SSIM and road coverage to estimate construction progress
    progress_percent = (1 - similarity_score) * 100 + road_coverage_change_percent

    # Display images and comparison results
    cv2.imshow("Reference Image", resized_image1)
    cv2.imshow("Test Image", resized_image2)
    cv2.imshow("Edge Detection Reference", edges_image1)
    cv2.imshow("Edge Detection Test", edges_image2)
    cv2.imshow("Keypoint Matches", img_matches)

    # Wait for keypress and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print additional parameters
    print(f"Number of Keypoints in Reference Image: {len(kp1)}")
    print(f"Number of Keypoints in Test Image: {len(kp2)}")
    print(f"Number of Matches: {len(matches)}")
    print(f"Road Pixels in Reference Image (edges): {np.sum(edges_image1 > 0)}")
    print(f"Road Pixels in Test Image (edges): {np.sum(edges_image2 > 0)}")
    print(f"Histogram Correlation (1.0 = perfect match): {hist_diff:.2f}")
    print(f"Utility Ducts Change (lines detected): {utility_ducts_change}")

    # Return analysis results
    return {
        "similarity_score": similarity_score * 100,  # Convert to percentage
        "road_coverage_change_percent": road_coverage_change_percent,
        "pixel_intensity_difference": pixel_diff,
        "histogram_difference": hist_diff,
        "utility_ducts_change": utility_ducts_change,
        "estimated_construction_progress_percent": progress_percent
    }

# Paths to the input images (replace with your own paths)
image_path1 = "first.jpg"
image_path2 = "secound.jpg"

# Calculate the progress between two images
progress_result = calculate_construction_progress(image_path1, image_path2)

# Print the result
print(f"Similarity Score : {progress_result['similarity_score']:.2f}%")
print(f"Road Coverage Change: {progress_result['road_coverage_change_percent']:.2f}%")
print(f"Pixel Intensity Difference: {progress_result['pixel_intensity_difference']:.2f}")
print(f"Histogram Difference: {progress_result['histogram_difference']:.2f}")
print(f"Utility Ducts Change: {progress_result['utility_ducts_change']}")
print(f"Estimated Construction Progress: {progress_result['estimated_construction_progress_percent']:.2f}%")