import cv2
import numpy as np
from scipy.interpolate import splprep, splev

# Step 1: Preprocessing
image = cv2.imread('test1.png', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Edge Detection
edges = cv2.Canny(blurred, 50, 150)

# Step 3: Contour Detection
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Define the threshold for arc length
threshold = 100  # You can adjust this value based on your data

# Step 4: Analyze Contours and Identify Gaps
incomplete_curves = [c for c in contours if cv2.arcLength(c, True) > threshold]

# Step 5: Check if we have incomplete curves
if len(incomplete_curves) > 0:
    # Assuming only one curve to simplify
    curve = incomplete_curves[0]
    curve = curve.squeeze()  # Removing extra dimensions

    # Step 6: Curve Completion using Spline Interpolation
    tck, u = splprep(curve.T, s=0)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck)

    completed_curve = np.vstack((x_new, y_new)).T.astype(int)

    # Step 7: Integrate the completed curve into the original image
    output_image = image.copy()
    for point in completed_curve:
        cv2.circle(output_image, tuple(point), 1, 255, -1)

    # Step 8: Save the output
    cv2.imwrite('test1output.png', output_image)
else:
    print("No incomplete curves found.")
