import cv2
import numpy as np

# Load the image
image_path = "test.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Threshold the image to get a binary image
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours with hierarchy
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Create an output image to draw regularized shapes
output = np.zeros_like(image)

# Loop over the contours to identify and regularize shapes
for i, contour in enumerate(contours):
    # Ignore small contours to reduce noise
    if cv2.contourArea(contour) < 100:
        continue

    # Calculate the perimeter of the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

    # Check if the shape is approximately a rectangle
    if len(approx) == 4 and hierarchy[0][i][3] == -1:
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Check if the shape is approximately circular
    elif len(approx) > 4:
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circularity = 4 * np.pi * (area / (peri * peri))

        # Check if the shape is close to a circle
        if 0.7 < circularity <= 1.2:
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output, center, radius, (255, 255, 255), 2)
        else:
            # Skip if the shape is too irregular to regularize
            continue

# Show the original image and the output with regularized shapes
cv2.imshow("Original Image", image)
cv2.imshow("Regularized Shapes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
