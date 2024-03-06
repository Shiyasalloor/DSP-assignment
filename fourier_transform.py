import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_spectrum(f_transform, title, energy):
    spectrum = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(spectrum))
    
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title(f'Original Image\nEnergy: {energy[0]:.2f}'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f'{title}\nEnergy: {energy[1]:.2f}'), plt.xticks([]), plt.yticks([])

    plt.show()

# Load the image in grayscale with an absolute file path
img = cv2.imread(r'C:/Users/shiya/Desktop/study works/DSP-assignment/dog.png', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if img is None:
    print("Error: Unable to load the image.")
else:
    # Calculate energy of the original image
    energy_original = np.sum(np.abs(img.astype(np.float64))**2)

    # Apply 2D Fourier Transform
    f_transform = np.fft.fft2(img)

    # Shift zero frequency components to the center
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Calculate energy of the Fourier-transformed image
    energy_transformed = np.sum(np.abs(f_transform_shifted)**2)

    # Normalize the Fourier-transformed image energy to be approximately the same as the original image energy
    scale_factor = energy_original / energy_transformed
    f_transform_shifted_normalized = f_transform_shifted * np.sqrt(scale_factor)

    # Display the original and Fourier Transformed images with energy
    plot_spectrum(f_transform_shifted_normalized, 'Fourier Transform (Normalized)', (energy_original, energy_transformed))
