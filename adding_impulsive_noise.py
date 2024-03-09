import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("./dog.png")

# Add impulsive noise (salt-and-pepper noise)
prob = 0.05  # Adjust this parameter to control the noise density
impulsive_noise = np.random.random(image.shape[:2])
image_with_impulsive_noise = image.copy()
image_with_impulsive_noise[impulsive_noise < prob / 2] = 0  # Set to black (0)
image_with_impulsive_noise[impulsive_noise > 1 - prob / 2] = 255  # Set to white (255)

plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(122), plt.imshow(cv2.cvtColor(image_with_impulsive_noise, cv2.COLOR_BGR2RGB)), plt.title("Impulsive Noise")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
