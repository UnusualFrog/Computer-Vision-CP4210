import cv2
import numpy as np


def transform_img(image):
    height, width = image.shape[:2]
    quarter_height, quarter_width = height / 4, width / 4

    T = np.float32([
        [1, 0, quarter_width],
        [0, 1, quarter_height]
        ])

    img_translation = cv2.warpAffine(image, T, (width, height))

    return img_translation

def reflect_img(image):
    height, width = image.shape[:2]

    T = np.float32([
        [-1, 0, height],
        [0, -1, width]
    ])
    
    img_reflection = cv2.warpAffine(image, T, (width, height))

    # 3D transform
    # T = np.float32([
    #     [1, 0, 0],
    #     [0, -1, height-1],
    #     [0, 0 , 1]
    #                 ])

    # img_reflection = cv2.warpPerspective(image, T, (width, height))

    return img_reflection

def rotate_img(image):
    height, width = image.shape[:2]
    
    rotation_matrix = cv2.getRotationMatrix2D((height/2, width/2) , 90, 1.0)
    
    img_rotation = cv2.warpAffine(image, rotation_matrix, (height, width))

    return img_rotation

def scale_img(image):
    height, width = image.shape[:2]
    
    img_shrinked = cv2.resize(image, (int(height/5), int(width/5)), interpolation=cv2.INTER_AREA)
    
    return img_shrinked

def shear_img(image):
    height, width = image.shape[:2]
    M = np.float32([
        [1, 0.5, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    sheared_img = cv2.warpPerspective(image, M, (height, width))
    
    return sheared_img

def image_enhance(image):
    contrast = 1.0
    brightness = 0
    
    img_adjusted = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
    
    return img_adjusted

def image_invert(image):
    inverted_img = 255- image
    
    return inverted_img

def hist_eq(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Actual Result
    image_eq = cv2.equalizeHist(grayscale)
    
    # Comparison Image
    res = np.hstack((grayscale, image_eq))
    
    # return image_eq
    return res

def image_sharpen(image):
    kernel_sharp = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    
    img_sharp = cv2.filter2D(image, -1, kernel_sharp)
    return img_sharp

def color_adj(image):
    
    B, G, R = cv2.split(image)
    
    return B, G, R

def image_denoise(image):
    sigma = 1.5
    
    denoise_image = cv2.GaussianBlur(image, (3,3), sigma)
    
    return denoise_image
    


def main():
    img = cv2.imread("ParrotImage.webp")
    # img = cv2.imread("Sample.jpg")
    
    # translated_img = transform_img(img)
    # cv2.imshow('img_transform', translated_img)
    # cv2.waitKey(0)
    
    # reflected_img = reflect_img(img)
    # cv2.imshow('img_reflect', reflected_img)
    # cv2.waitKey(0)
    
    # rotated_img = rotate_img(img)
    # cv2.imshow('img_rotate', rotated_img)
    # cv2.waitKey(0)
    
    # scaled_img = scale_img(img)
    # cv2.imshow('img_scaled', scaled_img)
    # cv2.waitKey(0)
    
    # sheared_img = shear_img(img)
    # cv2.imshow('img_shear', sheared_img)
    # cv2.waitKey(0)
    
    # adjusted_img = image_enhance(img)
    # cv2.imshow('img_adj', adjusted_img)
    # cv2.waitKey(0)
    
    # inverted_img = image_invert(img)
    # cv2.imshow('img_invert', inverted_img)
    # cv2.waitKey(0)
    
    # eq_img = hist_eq(img)
    # cv2.imshow('img_hist_eq', eq_img)
    # cv2.waitKey(0)

    # sharp_img = image_sharpen(img)
    # cv2.imshow('img_sharp', sharp_img)
    # cv2.waitKey(0)
    
    # color_adj_img_blue, color_adj_img_green, color_adj_img_red  = color_adj(img)
    # cv2.imshow('img_color_adj_blue', color_adj_img_blue)
    # cv2.waitKey(0)
    # cv2.imshow('img_color_adj_green', color_adj_img_green)
    # cv2.waitKey(0)
    # cv2.imshow('img_color_adj_red', color_adj_img_red)
    # cv2.waitKey(0)
    
    denoise_image = image_denoise(img)
    cv2.imshow('img_denoise', denoise_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
