# Gunawardhana P.K.K.C.
# EG/2019/3596
# Task 02

import cv2
import numpy as np

def grow_region(image, seeds, threshold, delay):
    # Defining 8-connectivity
    connectivity = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Creating a mask to keep track of visited pixels
    mask = np.zeros_like(image, dtype=np.uint8)
    h, w = image.shape[:2]

    # Creating a queue to store pixels to visit
    queue = list(seeds)

    # Segmenting the image
    segmented_image = np.zeros_like(image, dtype=np.uint8)

    while queue:
        current_pixel = queue.pop(0)
        x, y = current_pixel

        segmented_image[x, y] = image[x, y]

        mask[x, y] = 255

        # Checking the neighbors
        for dx, dy in connectivity:
            nx, ny = x + dx, y + dy

            # Check if neighbor is within bounds and unvisited
            if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] == 0:
                # Check if neighbor's intensity is within threshold
                if abs(int(image[nx, ny]) - int(image[x, y])) <= threshold:
                    queue.append((nx, ny))
                    mask[nx, ny] = 255

        # Display the current state of the segmented image
        #cv2.imshow('Segmented Image', segmented_image)
        #cv2.waitKey(delay)

    return segmented_image

# Loading image
input_image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)

# Defining seeds (start points)
# with 2 seeds
seeds = [(230, 40),(200,350)]
# with only 1 seed
#seeds = [(162, 162)]

# Ensure seeds are within image bounds
seeds = [(min(max(p[0], 0), input_image.shape[0] - 1), min(max(p[1], 0), input_image.shape[1] - 1)) for p in seeds]

# Setting the threshold for region growing
threshold = 2

delay = 1

# Performing the region growing
segmented_image = grow_region(input_image, seeds, threshold, delay)

# Display final result
cv2.imshow('Original Image', input_image)
cv2.imshow('Final Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
