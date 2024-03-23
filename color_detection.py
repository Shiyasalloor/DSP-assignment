import cv2
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the color of the pixel at the clicked location
        b, g, r = image[y, x]
        print(f"Color at pixel ({x}, {y}): R={r}, G={g}, B={b}")

# Load the image
image_path = "./dog.png"
image = cv2.imread(image_path)

# Create a window and set mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Display the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
