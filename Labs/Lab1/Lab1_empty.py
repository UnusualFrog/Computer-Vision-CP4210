import cv2

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

def SIFT_Helper(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    return cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


#h_factor = how much should the image's height be scaled
#w_factor = how much should the image's width be scaled
def SIFT(h_factor, w_factor):
    # We want to match the features between two images
    # read the images 'book.jpg' and 'table.jpg'
    # convert images to grayscale
    img1 = cv2.imread("book.jpg", 0)
    img2 = cv2.imread("table.jpg", 0)

    # cv2.imshow("img1", img1)
    # cv2.waitKey(0)
    # cv2.imshow("img2", img2)
    # cv2.waitKey(0)
    
    # Scale the image in accordance to the arguments being passed in (h_factor, w_factor)
    img1_scaled = cv2.resize(img1, (h_factor, w_factor))
    img2_scaled = cv2.resize(img2, (h_factor, w_factor))

    # cv2.imshow("img1_scaled", img1_scaled)
    # cv2.waitKey(0)
    # cv2.imshow("img2_scaled", img2_scaled)
    # cv2.waitKey(0)

    # Check how well the SIFT technique performs when you rescale the image by various amounts
    img1_scaled_wide = cv2.resize(img1, (h_factor * 2, w_factor))
    img2_scaled_wide = cv2.resize(img1, (h_factor * 2, w_factor))

    img1_scaled_tall = cv2.resize(img1, (h_factor, w_factor * 2))
    img2_scaled_tall = cv2.resize(img1, (h_factor, w_factor * 2))
    
    #Use SIFT to match the features between the two images
    #I recommend you use two built-in objects (see online documentation)
    # - SIFT
    # - BFMatcher
    # NOTE: helper function added above for reusing SIFT matching
    img3 = SIFT_Helper(img1_scaled, img2_scaled)
    img3_tall = SIFT_Helper(img1_scaled_tall, img2_scaled_tall)
    img3_wide = SIFT_Helper(img1_scaled_wide, img2_scaled_wide)

    # show the image
    # NORMAL SCALE
    cv2.imshow("img3", img3)
    cv2.waitKey(0)

    # TALL SCALE
    cv2.imshow("img3_tall", img3_tall)
    cv2.waitKey(0)

    # WIDE SCALE
    cv2.imshow("img3_wide", img3_wide)
    cv2.waitKey(0)

def HOG():
    # Download a pre-existing image from the scikit-image toolbox
    image = data.astronaut() #Color image of the astronaut Eileen Collins

    
    #Rotate the image by however many degrees you choose
    rotated_img = cv2.rotate(image, cv2.ROTATE_180)

    # Use HOG technique
    fd, hog_image = hog(
        rotated_img,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(rotated_img, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    # Visualize the results
    plt.show()


    
    
def FAST():
    # Load the image ('book.jpg')
    img1 = cv2.imread("book.jpg", cv2.IMREAD_GRAYSCALE)

    #Rotate the image by however many degrees you choose
    rotated_img = cv2.rotate(img1, cv2.ROTATE_180)

    cv2.imshow("rotated_img", rotated_img)
    cv2.waitKey(0)
    
    # Create a FAST object
    # You can specify parameters like threshold, non-maximum suppression (NMS) status, etc.
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(rotated_img, None)
    img2 = cv2.drawKeypoints(rotated_img, kp, None, color=(255,0,0))

    # Display the result (example, requires a display environment).
    # HINT: You can use a function (see online documentation for more info) inside the 'exposure' library (that is imported in the beginning of the code) to rescale the intensity of the result for better display.
    cv2.imshow('img2', img2)
    cv2.waitKey(0)

def main():
    # Call the above three functions. Keep in mind that SIFT(...) requires two arguments.
    h_factor = 512; w_factor = 512
    # SIFT(h_factor, w_factor)
    # HOG()
    FAST()
    

if __name__ == "__main__":
    main()
