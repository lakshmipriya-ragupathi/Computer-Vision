import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("photo.jpg")
image = image[1800:2125, 1410:1750]
image = cv2.GaussianBlur(image, (5, 5), 5)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sharpening_factor = 0.5


sigma = 1.5
image_smoothed = cv2.GaussianBlur(gray_image, (0, 0), sigma)
image_log = cv2.Laplacian(image_smoothed, cv2.CV_64F)
image_log = cv2.normalize(image_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
threshold_value = 150
edges = cv2.threshold(image_log, threshold_value, 255, cv2.THRESH_BINARY)[1]
sharpened_image = cv2.addWeighted(
    image,
    1,
    cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
    sharpening_factor,
    0,
)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.title("Marr Hildreth sharpened image")
plt.show()
