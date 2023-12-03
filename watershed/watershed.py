
#Convert to binary image, foreground is white and background is black

#find sure foreground 

#find sure background

#for unlabelled pixel,
#find connected components in sure fg image
#  apply watershed algorithm
#find pixels with local minima's, connected components in the image
#for each local minima assign a distinct label
#for each intensity k to Max
#   for each unlabelled pixel, if p adjacent to signle label, assign the label, foreground
#                              else dam point or background
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

image = cv2.imread("image.jpg")
#image = image[1800:2125, 1410:1750]
image = image[0:3000, ]
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#plt.imshow(image, cmap='gray')
#plt.title("image")
#plt.show()

#Convert to binary image, foreground is white and background is black
inverted_image = cv2.bitwise_not(image1)
ret, binary = cv2.threshold(inverted_image, 130,255,0)

#opening operation to remove noise:
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

#dialte for sure background:
#find sure background 
kernel2 = np.ones((5, 5), np.uint8) 
sure_bg = cv2.dilate(opening, kernel2, iterations=1) 

#for sure foreground:
sure_fg = cv2.erode(opening, kernel, iterations=1) 

unknown = cv2.subtract(sure_bg, sure_fg)

#find connected components for local minimas for watershed
rettt, markers = cv2.connectedComponents(sure_fg)
print("number of connected components =", rettt)

#perform watershed
markers = cv2.watershed(image, markers)

# Create a custom color mapping for visualization
segmented_image = np.zeros_like(image)
for label in np.unique(markers):
    if label == 0:
        continue  # Skip the background
    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    segmented_image[markers == label] = color


plt.imshow(image)
plt.title("original image")
plt.show()


plt.imshow(binary, cmap='gray')
plt.title("binary")
plt.show()

plt.imshow(opening, cmap='gray')
plt.title("opened")
plt.show()

plt.imshow(sure_bg, cmap='gray')
plt.title("dilated")
plt.show()

plt.imshow(sure_fg, cmap='gray')
plt.title("eroded")
plt.show()

plt.imshow(unknown, cmap='gray')
plt.title("unknown")
plt.show()

plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title("Segmented Image")
plt.show()


