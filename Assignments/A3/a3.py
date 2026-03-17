import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("kings_hand.jpg")

image_color_corrected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply grayscale and gaussian blur to reduce noise 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray_image, (3, 3), 0)

#  Prewitt for horizontal edge detection
prewitt_x = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

#  Prewitt for vertical edge detection
prewitt_y = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

# Apply horizontal edge detection 
img_prewittx = cv2.filter2D(img_gaussian, -1, prewitt_x)

# Apply vertical edge detection
img_prewitty = cv2.filter2D(img_gaussian, -1, prewitt_y)

# Combine vertical and horizontal output
img_prewitt = np.sqrt(img_prewittx.astype(np.float32)**2 + img_prewitty.astype(np.float32)**2)

# Convert the image back to uint8
img_prewitt_uint8 = cv2.convertScaleAbs(img_prewitt)

# Compare the original image to the Prewitt image
fig, ax = plt.subplots(1,2)
ax[0].imshow(image_color_corrected)
ax[0].set_title('Original Image')

ax[1].imshow(img_prewitt_uint8, cmap='gray')
ax[1].set_title('Prewitt Edge Detection')
plt.show()