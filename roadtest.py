import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity as ssim

# Preprocessing function to resize, grayscale, and normalize images
def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to standard size (let's use 512x512)
    resized_image = cv2.resize(gray_image, (512, 512))
    # Normalize
    normalized_image = resized_image / 255.0
    return normalized_image

# Function to compute edge detection (helps identify road surface changes)
def edge_detection(image):
    # Apply Canny edge detector
    edges = cv2.Canny((image * 255).astype(np.uint8), threshold1=100, threshold2=200)
    return edges

# Function to compute similarity score using Structural Similarity Index (SSIM)
def compute_similarity(image1, image2):
    # Since the images are normalized to [0, 1], we set data_range to 1.0
    similarity, _ = ssim(image1, image2, full=True, data_range=1.0)
    return similarity

# Function to calculate road coverage percentage based on edge detection
def road_coverage_change(edge_image1, edge_image2):
    road_pixels1 = np.sum(edge_image1 > 0)
    road_pixels2 = np.sum(edge_image2 > 0)
    # Calculate percentage increase in visible road area (based on edges)
    if road_pixels1 == 0:  # To avoid division by zero
        return 0
    return (road_pixels2 - road_pixels1) / road_pixels1 * 100

# Main function to process two images and calculate construction progress
def calculate_construction_progress(image_path1, image_path2):
    # Preprocess both images
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)

    # Detect edges in both images
    edges_image1 = edge_detection(image1)
    edges_image2 = edge_detection(image2)

    # Calculate SSIM (Structural Similarity Index) to measure overall similarity
    similarity_score = compute_similarity(image1, image2)

    # Calculate road coverage change based on edges
    road_coverage_change_percent = road_coverage_change(edges_image1, edges_image2)

    # Combining SSIM and road coverage to estimate construction progress
    progress_percent = (1 - similarity_score) * 100 + road_coverage_change_percent

    # Return analysis results
    return {
        "similarity_score": similarity_score * 100,  # Convert to percentage
        "road_coverage_change_percent": road_coverage_change_percent,
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
print(f"Estimated Construction Progress: {progress_result['estimated_construction_progress_percent']:.2f}%")