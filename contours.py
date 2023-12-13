import cv2
import numpy as np

# Read an image
image = cv2.imread("IMG_20231213_152313.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Display the result
cv2.imwrite("Contours.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
