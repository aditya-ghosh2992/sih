import cv2
import numpy as np

def detect_anomaly(reference_image_path, test_image_path, threshold=0.8):
    # Read the reference and test images in grayscale
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded correctly
    if reference_image is None or test_image is None:
        print("Error loading images.")
        return

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(reference_image, None)
    kp2, des2 = sift.detectAndCompute(test_image, None)

    # Use BFMatcher to find matches between descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate the anomaly score
    anomaly_score = 1 - len(good_matches) / len(kp1)

    # Print anomaly status
    if anomaly_score > threshold:
        print("Anomaly Detected!")
    else:
        print("No Anomaly Detected.")

    # Draw matches on the images
    img_matches = cv2.drawMatches(reference_image, kp1, test_image, kp2, good_matches, None)

    # Resize images for better visualization
    resized_reference_image = cv2.resize(reference_image, (400, 300))
    resized_test_image = cv2.resize(test_image, (400, 300))
    resized_img_matches = cv2.resize(img_matches, (800, 600))

    # Display the images
    cv2.imshow("Resized Reference Image", resized_reference_image)
    cv2.imshow("Resized Test Image", resized_test_image)
    cv2.imshow("Resized Matches", resized_img_matches)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display non-matching keypoints (for visualization, if needed)
    non_matching_kp2 = [kp2[m.trainIdx] for m in matches if m not in good_matches]
    img_non_matching = cv2.drawKeypoints(test_image, non_matching_kp2, None, color=(0, 0, 255))
    resized_img_non_matching = cv2.resize(img_non_matching, (400, 300))

    cv2.imshow("Resized Non-Matching Keypoints", resized_img_non_matching)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Paths to the reference and test images
reference_image_path = "first.jpg"
test_image_path = "secound.jpg"
detect_anomaly(reference_image_path, test_image_path)