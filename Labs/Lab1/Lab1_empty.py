# Import relevant libraries
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

"""
Helper function to assist the SIFT function in performing BFM image matching

:param img1: first image to be used in matching
:param img2: second image to be used in matching
:return: image containing the matching keypoints between img1 and img2
"""
def SIFT_Helper(img1, img2):
    # Create the sift object
    sift = cv2.SIFT_create()
    # Find and save key points and descriptors for img1 & img2
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher()

    # Use knnMatcher to identfiy k best matching keypoints between the descriptors of img1 & img2
    # k=2 used as it is the default for knnMatch 
    matches = bf.knnMatch(des1, des2, k=2)

    # Perform ratio test on all matching keypoints between img1 & img2 to ensure
    # that only keypoints which have a meaningfully different distance 
    # are used in drawing matches between img1 & img2
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    # Draw and return a new image using img1 & img2, their respective keypoints and matching keypoints which pass the ratio test
    # New image drawn will contain both img1 & img2 side-by-side with a line drawn between matching keypoints in both images
    # the flag parameter ensures only keypoints with a match in both images will be drawn
    return cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


#INS. COMMENT h_factor = how much should the image's height be scaled
#INS. COMMENT w_factor = how much should the image's width be scaled
def SIFT(h_factor, w_factor):
    #INS. COMMENT We want to match the features between two images
    #INS. COMMENT read the images 'book.jpg' and 'table.jpg'
    #INS. COMMENT convert images to grayscale

    # Load the book and table images in grayscale using the flag 0 which is shorthand for load using grayscale
    img1 = cv2.imread("book.jpg", 0)
    img2 = cv2.imread("table.jpg", 0)
    
    #INS. COMMENT Scale the image in accordance to the arguments being passed in (h_factor, w_factor)

    # Scale images by the height and weight factor parameters
    img1_scaled = cv2.resize(img1, (h_factor, w_factor))
    img2_scaled = cv2.resize(img2, (h_factor, w_factor))

    #INS. COMMENT Check how well the SIFT technique performs when you rescale the image by various amounts

    # Alternate scaling - double the height of the images
    img1_scaled_wide = cv2.resize(img1, (h_factor * 2, w_factor))
    img2_scaled_wide = cv2.resize(img2, (h_factor * 2, w_factor))

    # Alternate scaling - double the width of the images
    img1_scaled_tall = cv2.resize(img1, (h_factor, w_factor * 2))
    img2_scaled_tall = cv2.resize(img2, (h_factor, w_factor * 2))
    
    #INS. COMMENT Use SIFT to match the features between the two images
    #INS. COMMENT I recommend you use two built-in objects (see online documentation)
    #INS. COMMENT  - SIFT
    #INS. COMMENT  - BFMatcher

    # Detect key points of img1 & img2 with SIFT and compare keypoints using BFMatcher
    # NOTE: helper function added above for reusing SIFT matching logic
    img3 = SIFT_Helper(img1_scaled, img2_scaled)
    img3_tall = SIFT_Helper(img1_scaled_tall, img2_scaled_tall)
    img3_wide = SIFT_Helper(img1_scaled_wide, img2_scaled_wide)

    #INS. COMMENT show the image
    # Display the image scaled by the height and width factors normally
    cv2.imshow("img3", img3)
    cv2.waitKey(0)

     # Display the image scaled by the double the height factor and regular width factor
    cv2.imshow("img3_tall", img3_tall)
    cv2.waitKey(0)

    # Display the image scaled by the double the width factor and regular height factor
    cv2.imshow("img3_wide", img3_wide)
    cv2.waitKey(0)

def HOG():
    #INS. COMMENT  Download a pre-existing image from the scikit-image toolbox
    image = data.astronaut() #Color image of the astronaut Eileen Collins

    #INS. COMMENT Rotate the image by however many degrees you choose
    # Rotate the image by 180 degrees, flipping it upside down
    rotated_img = cv2.rotate(image, cv2.ROTATE_180)

    #INS. COMMENT  Use HOG technique

    # Generate a hog image using rotated_img with 8 orientations, 16 pixels per cell, 1 cell per block, and use all RGB channels
    fd, hog_image = hog(
        rotated_img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
    )

    # Create a figure for comparing rotated_img to hog_image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(rotated_img, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    #INS. COMMENT  Visualize the results
    plt.show()
 
def FAST():
    #INS. COMMENT  Load the image ('book.jpg')

    # Load book.jpg in grayscale using the flag cv2.IMREAD_GRAYSCALE
    img1 = cv2.imread("book.jpg", cv2.IMREAD_GRAYSCALE)

    #INS. COMMENT Rotate the image by however many degrees you choose
    # Rotate img1 by 180 degrees to flip it upside down
    rotated_img = cv2.rotate(img1, cv2.ROTATE_180)
    
    #INS. COMMENT Create a FAST object
    #INS. COMMENT You can specify parameters like threshold, non-maximum suppression (NMS) status, etc.

    # Create FAST object
    fast = cv2.FastFeatureDetector_create()

    # Find keypoints in rotated_img
    kp = fast.detect(rotated_img, None)

    # Draw keypoints previously detected onto rotated_img, display the keypoints in red 
    # NOTE: due to openCV using BGR vs. matplotlib using RGB, the points will
    #  display as blue when displayed in the next step
    img2 = cv2.drawKeypoints(rotated_img, kp, None, color=(255,0,0))

    #INS. COMMENT Display the result (example, requires a display environment).
    #INS. COMMENT HINT: You can use a function (see online documentation for more info) inside the 'exposure' library (that is imported in the beginning of the code) to rescale the intensity of the result for better display.
    cv2.imshow('img2', img2)
    cv2.waitKey(0)

def main():
    #INS. COMMENT Call the above three functions. Keep in mind that SIFT(...) requires two arguments.
    h_factor = 512; w_factor = 512
    # SIFT(h_factor, w_factor)
    HOG()
    # FAST()
    

if __name__ == "__main__":
    main()
