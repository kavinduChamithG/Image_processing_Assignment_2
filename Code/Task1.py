# Gunawardhana P.K.K.C.
# EG/2019/3596
# Task 01

import numpy as np
import cv2
from matplotlib import pyplot as plt

def image_with_objects_generator(shape):
    #Defining the pixel values for each object anf the background
    background_pixel_value = 0
    object1_pixel_value = 200
    object2_pixel_value = 100

    # Create an empty image with background value
    image = np.full(shape, background_pixel_value, dtype=np.uint8)

    # Drawing a circle (object 1)
    cv2.circle(image, (128, 70), 40, object1_pixel_value, -1)

    # Drawing a triangle (object 2)
    points = np.array([[68, 150], [188, 150], [128, 230]], np.int32)
    cv2.fillPoly(image, [points], object2_pixel_value)

    return image

def gaussian_noise_adder(image):
    mean=0
    sigma=25
    #generating a gaussian noise with the defined mean and sigma
    noise = np.random.normal(mean, sigma, image.shape)
    
    #adding noise to the image and clip the values to the [0 , 255] range
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return noisy_image

def otsu_threshold_algorithm(image):
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded_image

# Generating the image with 2 objects
image_shape = (256, 256)
objects_image = image_with_objects_generator(image_shape)

# Adding Gaussian noise to the image
noisy_objects_image = gaussian_noise_adder(objects_image)

# Performing Otsu's thresholding
thresholded_objects_image = otsu_threshold_algorithm(noisy_objects_image)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(objects_image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(noisy_objects_image, cmap='gray')
plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(thresholded_objects_image, cmap='gray')
plt.title('Thresholded Image'), plt.xticks([]), plt.yticks([])
plt.show()
