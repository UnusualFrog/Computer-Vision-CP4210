# Import libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese

# Load image as grayscale
img_1 = cv.imread("Sample.jpg", 0)
img_2 = cv.imread("seasons.png", 0)


# plt.imshow(img_1, cmap="grey")
# plt.title('img_1')
# plt.show()

# plt.imshow(img_2)
# plt.title('img_2')
# plt.show()

# Apply image segmentation with chan-vesse
img_1_cv = chan_vese(
    img_1,
    max_num_iter=200,
    extended_output=True,
)

fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax = axes.flatten()

# plot original image
ax[0].imshow(img_1, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

# plot segemented image
ax[1].imshow(img_1_cv[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(img_1_cv[2])} iterations'
ax[1].set_title(title, fontsize=12)

# show before and after segementation
fig.tight_layout()
plt.show()

# Apply image segmentation with chan-vesse
img_2_cv = chan_vese(
    img_2,
    max_num_iter=200,
    extended_output=True,
)

fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax = axes.flatten()

# plot original image
ax[0].imshow(img_2, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

# plot segemented image
ax[1].imshow(img_2_cv[0], cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation - {len(img_2_cv[2])} iterations'
ax[1].set_title(title, fontsize=12)

# show before and after segementation
fig.tight_layout()
plt.show()
