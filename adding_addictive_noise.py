import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("./dog.png")

mean = 0
std_dev = 25
noise = np.random.normal(mean,std_dev,image.shape)

noisy_image = image + noise.astype(np.uint8)

plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(122), plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)), plt.title("Noisy")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()