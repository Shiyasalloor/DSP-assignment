import cv2
import numpy as np

image = cv2.imread("./dog.png")
cv2.imshow("image", image)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("grayscale image", grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

gradient_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)

gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

image_energy = np.sum(gradient_magnitude**2)

print(f"Image Energy: {image_energy}")