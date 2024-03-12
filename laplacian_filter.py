import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('./cat.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the Laplacian filter
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

# Apply the Laplacian filter to each color channel
laplacian_red = cv2.filter2D(image[:, :, 2], cv2.CV_64F, laplacian_kernel)
laplacian_green = cv2.filter2D(image[:, :, 1], cv2.CV_64F, laplacian_kernel)
laplacian_blue = cv2.filter2D(image[:, :, 0], cv2.CV_64F, laplacian_kernel)

# Combine the results 
laplacian_combined = np.uint8(np.absolute(laplacian_red) + np.absolute(laplacian_green) + np.absolute(laplacian_blue))

# Add the Laplacian filter to the original image
filtered_image = cv2.addWeighted(image, 1, cv2.merge([laplacian_combined]*3), 1, 0)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_combined, cmap='gray')
plt.title('Laplacian Filter')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Laplacian Filter Added')
plt.axis('off')

plt.show()
