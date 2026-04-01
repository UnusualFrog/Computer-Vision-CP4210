import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1 = cv2.imread("House1.jpg")
image2 = cv2.imread("House2.jpg")

# Convert images from BGR to RGB for plotting
image1_color_corrected = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_color_corrected = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Convert to float32 for Harris corner detection
gray1_float = np.float32(gray_image1)
gray2_float = np.float32(gray_image2)

# Apply Harris Corner Detector with window size 2, sobel size 3, and 0.04 harris count
corners1 = cv2.cornerHarris(gray1_float, 2, 3, 0.04)
corners2 = cv2.cornerHarris(gray2_float, 2, 3, 0.04)

# Dilate corner results to enhance corner points
corners1 = cv2.dilate(corners1, None)
corners2 = cv2.dilate(corners2, None)

# Threshold for detecting strong corners
threshold1 = 0.01 * corners1.max()
threshold2 = 0.01 * corners2.max()

# Feature Matching using ORB

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

# Create Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top matches
matched_image = cv2.drawMatches(
    image1_color_corrected, keypoints1,
    image2_color_corrected, keypoints2,
    matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Create copies for visualization
image1_corners = image1_color_corrected.copy()
image2_corners = image2_color_corrected.copy()

# Mark corners in red
image1_corners[corners1 > threshold1] = [255, 0, 0]
image2_corners[corners2 > threshold2] = [255, 0, 0]

# Plot Results
fig1, ax1 = plt.subplots(1, 2)

ax1[0].imshow(image1_corners)
ax1[0].set_title('House1 - Harris Corners')

ax1[1].imshow(image2_corners)
ax1[1].set_title('House2 - Harris Corners')

plt.show()

# Show feature matching result
plt.figure()
plt.imshow(matched_image)
plt.title('Feature Matching (ORB + Brute Force)')
plt.axis('off')
plt.show()