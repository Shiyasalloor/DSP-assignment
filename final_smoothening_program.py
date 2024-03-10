import cv2
import numpy as np
from matplotlib import pyplot as plt

def add_noise(image, noise_type='additive', prob=0.05, mean=0, std_dev=50):
    if noise_type == 'additive':
        noise = np.random.normal(mean, std_dev, image.shape)
        noisy_image = image + noise.astype(np.uint8)
    elif noise_type == 'impulsive':
        impulsive_noise = np.random.random(image.shape[:2])
        noisy_image = image.copy()
        noisy_image[impulsive_noise < prob / 2] = 0 
        noisy_image[impulsive_noise > 1 - prob / 2] = 255 
    else:
        raise ValueError("Invalid noise_type. Choose 'additive' or 'impulsive'")
    return noisy_image

def apply_filter(noisy_image, filter_type='average', kernel_size=5):
    if filter_type == 'average':
        filter_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        smoothed_image = cv2.filter2D(noisy_image, -1, filter_kernel)
    elif filter_type == 'median':
        smoothed_image = cv2.medianBlur(noisy_image, kernel_size)
    else:
        raise ValueError("Invalid filter_type. Choose 'average' or 'median'")
    return smoothed_image

# Load the original image
image = cv2.imread("./dog.png")

# Ask user for noise type
noise_type = input("Enter noise type ('additive' or 'impulsive'): ")

# Add noise based on user input
noisy_image = add_noise(image, noise_type=noise_type)

# Display the original and noisy images
plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(132), plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)), plt.title("Noisy")
plt.show()

# Ask user for filter type
filter_type = input("Enter filter type ('average' or 'median'): ")

# Ask user for filter size
kernel_size = int(input("Enter kernel size (an integer): "))

# Apply filter based on user input
smoothed_image = apply_filter(noisy_image, filter_type=filter_type, kernel_size=kernel_size)

# Display the noisy and smoothed images
plt.subplot(131), plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)), plt.title("Noisy")
plt.subplot(132), plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB)), plt.title("Smoothed")
plt.show()

# Calculate and print PSNR value
mse = np.mean((image - smoothed_image) ** 2)
max_pixel = 255.0
psnr = 10 * np.log10((max_pixel ** 2) / mse)
print(f"PSNR: {psnr} dB")

cv2.waitKey(0)
cv2.destroyAllWindows()
