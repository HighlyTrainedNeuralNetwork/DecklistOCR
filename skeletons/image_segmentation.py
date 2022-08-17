import cv2
import numpy as np

image = cv2.imread('assets/Grixis Phoenix Decklist.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLinesP(
    edges,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=200,  # Min number of votes for valid line
    minLineLength=5,  # Min allowed length of line
    maxLineGap=10  # Max allowed gap between line for joining them
)

vertical_lines = sorted(lines, key=lambda x: abs(x[0][0] - x[0][2]), reverse=True)
horizontal_lines = sorted(lines, key=lambda x: abs(x[0][1] - x[0][3]), reverse=True)
vertical_lines = vertical_lines[:5]
horizontal_lines = horizontal_lines[:5]
lines = vertical_lines + horizontal_lines


for points in lines:
    x1, y1, x2, y2 = points[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('assets/detectedLines.png', image)
