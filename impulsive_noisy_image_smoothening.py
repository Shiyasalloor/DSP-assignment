import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("./dog.png")

prob = 0.10  
impulsive_noise = np.random.random(image.shape[:2])
image_with_impulsive_noise = image.copy()
image_with_impulsive_noise[impulsive_noise < prob / 2] = 0  
image_with_impulsive_noise[impulsive_noise > 1 - prob / 2] = 255 

kernel_size = 3  
smoothed_image = cv2.medianBlur(image_with_impulsive_noise, kernel_size)

plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(132), plt.imshow(cv2.cvtColor(image_with_impulsive_noise, cv2.COLOR_BGR2RGB)), plt.title("Impulsive Noise")
plt.subplot(133), plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB)), plt.title("Smoothed")
plt.show()

mse = np.mean((image - smoothed_image) ** 2)
max_pixel = 255.0
psnr = 10 * np.log10((max_pixel ** 2) / mse)
print(f"PSNR: {psnr} dB")

cv2.waitKey(0)
cv2.destroyAllWindows()
