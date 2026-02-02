import cv2 as cv
import numpy as np
import skimage

def imgFilter():
    # INS. COMMENT Load the image 'Sample.jpg'
    img = cv.imread("Sample.jpg")

    # Use CV_16S to perserve negative gradient values
    image_depth = cv.CV_16S

    # INS. COMMENT Create the kernel used for image sharpening

    # === Laplacian sharpening kernel === 
    kernel_sharp_laplacian = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
        ])
    
    # === High-passs sharpening kernel === 
    kernel_sharp_high_pass = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    # ===  Unsharp Mask sharpening kernel === 
    kernel_sharp_unsharp_mask = np.array([
        [-1, -2, -1],
        [-2, 13, -2],
        [-1, -2, -1],
    ])
    
    #NOTE: Uncomment desired sharpening kernel
    # img_sharp = cv.filter2D(img, image_depth, kernel_sharp_laplacian)
    img_sharp = cv.filter2D(img, image_depth, kernel_sharp_high_pass)
    # img_sharp = cv.filter2D(img, image_depth, kernel_sharp_unsharp_mask)

    # cv.imshow('sharpend', img_sharp)
    # cv.waitKey(0)

    # INS. COMMENT Create the kernel used for edge detection
    
    # === Sobel edge detection === 
    # Calculate x and y derivatives for image
    grad_x = cv.Sobel(img_sharp, image_depth, 1, 0)
    grad_y = cv.Sobel(img_sharp, image_depth, 0, 1)

    # Combine gradient magnitude values to combine x and y gradients to 1 image
    img_grad = np.sqrt(grad_x.astype(float)**2, grad_y.astype(float)**2)

    # Handle signed gradient values 
    img_sobel = cv.convertScaleAbs(img_grad)
        
    # cv.imshow("Sobel", img_sobel)
    # cv.waitKey(0)

    # ===  Prewitt edge detection === 
    kernel_prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    kernel_prewitt_y = np.array([
        [-1,-1,-1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    # Apply kernels 
    img_prewitt_x = cv.filter2D(img_sharp, image_depth, kernel_prewitt_x)
    img_prewitt_y = cv.filter2D(img_sharp, image_depth, kernel_prewitt_y)

    # combine and handle signed gradient values
    img_magnitude = np.sqrt(img_prewitt_x.astype(float)**2, img_prewitt_y.astype(float)**2)

    img_prewitt = cv.convertScaleAbs(img_magnitude)

    # cv.imshow("Prewitt", img_prewitt)
    # cv.waitKey(0)

    # === Laplacian edge detection ===
    # Blur the image to remove noise as laplacian is sensitive to noise
    blurred = cv.GaussianBlur(img_sharp, (5,5), 0)

    # Apply the laplacian function with CV_64F to handle negative values
    laplacian = cv.Laplacian(blurred, cv.CV_64F)

    # Handle negative values
    laplacian_abs = np.absolute(laplacian)

    # Convert back to original image format for display
    img_laplacian = np.uint8(laplacian_abs)

    # cv.imshow("Laplacian", img_laplacian)
    # cv.waitKey(0)

    #NOTE: Uncomment for desired result
    # result = img_sobel
    result = img_prewitt
    # result = img_laplacian
    
    return result

def imgMorph():
    # INS. COMMENT Load the image 'Sample.jpg'
    img = cv.imread("Sample.jpg")

    # Create kernels
    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_5 = np.ones((5, 5), np.uint8)
    kernel_10 = np.ones((10, 10), np.uint8)

    # NOTE: Switch this variable to change kernel used by morphology
    current_kernel = kernel_5

    # INS. COMMENT Erode the image
    img_erode = cv.erode(img, current_kernel, iterations=1)
    # cv.imshow("Erode", img_erode)
    # cv.waitKey(0)

    # INS. COMMENT Dilate the image
    img_dilate = cv.dilate(img, current_kernel, iterations=1)
    # cv.imshow("Dilate", img_dilate)
    # cv.waitKey(0)

    # INS. COMMENT Perform image opening by using the built-in morphologyEx(...) in opencv
    img_open = cv.morphologyEx(img, cv.MORPH_OPEN, current_kernel)
    # cv.imshow("Opening", img_open)
    # cv.waitKey(0)

    # INS. COMMENT Perform image opening by using the built-in erode(...) and dilate(...) in opencv
    img_open_manual = cv.dilate(cv.erode(img, current_kernel, iterations=1), current_kernel, iterations=1)
    # cv.imshow("Opening with manual steps", img_open_manual)
    # cv.waitKey(0)

    # INS. COMMENT Perform image closing by using the built-in morphologyEx(...) in opencv
    img_close = cv.morphologyEx(img, cv.MORPH_CLOSE, current_kernel)
    cv.imshow("Close", img_close)
    cv.waitKey(0)

    # INS. COMMENT Perform image closing by using the built-in erode(...) and dilate(...) in opencv
    img_close_manual = cv.erode(cv.dilate(img, current_kernel, iterations=1), current_kernel, iterations=1)
    cv.imshow("Closing with manual steps", img_close_manual)
    cv.waitKey(0)

    return

def imgHist():
    # INS. COMMENT Perform equalized histogram on the original image using the built-in equalizeHist(...) in opencv

    # INS. COMMENT Try to get the equalized histogram on the original image using the built-in match_histograms(...) in skimage

    return

def main():
    result_filter = imgFilter()
    # cv.imshow("imgFilter Result", result_filter)
    # cv.waitKey(0)

    result_morph = imgMorph()
    cv.imshow("imgMorph Result", result_morph)
    cv.waitKey(0)

    result_hist = imgHist()
    # cv.imshow("imgHist Result", result_hist)
    # cv.waitKey(0)

if __name__ == "__main__":
    main()
