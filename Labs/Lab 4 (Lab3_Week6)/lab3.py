# INS. Import the necessary libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def low_pass(img, fft):
    # Generate a mask for a low pass filtering
    # get rows and column size from image
    x, y = img.shape
    # get centers of rows and columns
    cx, cy = x // 2, y // 2
    # create a mask of zeroes based on the image rows and columns
    mask_low = np.zeros((x, y), np.uint8)
    # populate mask based on center row and column shift 
    mask_low[cx-30:cx+30, cy-30:cy+30] = 1

    # Apply low-pass filter mask
    fft_masked_low = fft * mask_low
    # Apply inverse FFT to convert back to spatial domain
    in_fft_low = np.fft.ifftshift(fft_masked_low)
    image_filtered_low_pass = np.fft.ifft2(in_fft_low)
    image_filtered_low_pass = np.abs(image_filtered_low_pass)

    # Show image
    plt.imshow(image_filtered_low_pass, cmap='gray')
    plt.title('low pass filter')
    plt.show()

    return image_filtered_low_pass

def high_pass(img, fft):
    # Generate a mask for a high pass filtering
    # get rows and column size from image
    x, y = img.shape
    # get centers of rows and columns
    cx, cy = x // 2, y // 2
    # create a mask of ones based on the image rows and columns
    mask_high = np.ones((x, y), np.uint8)
    # populate mask based on center row and column shift 
    mask_high[cx-30:cx+30, cy-30:cy+30] = 0

    # Apply high-pass filter mask
    fft_masked_high = fft * mask_high
    # Apply inverse FFT to convert back to spatial domain
    in_fft_high = np.fft.ifftshift(fft_masked_high)
    image_filtered_high_pass = np.fft.ifft2(in_fft_high)
    image_filtered_high_pass = np.abs(image_filtered_high_pass)

    # Show image
    plt.imshow(image_filtered_high_pass, cmap='gray')
    plt.title('high pass filter')
    plt.show()

    return image_filtered_high_pass

def main():
    # INS. Load the image and convert it into grayscale
    img = cv.imread("Sample.jpg", 0)
 
    # INS. Conduct frequency-domain operations on the image:
    # Compute FFT
    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)

    # INS. Low-pass filtering
    low_pass_img = low_pass(img, fft)

    # INS. High-pass filtering
    high_pass_img = high_pass(img, fft)

if __name__ == "__main__":
    main()
