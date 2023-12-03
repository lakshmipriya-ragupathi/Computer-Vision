import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

#input : photo
#output sharpened image

#canny edge detection 


#loading the image
ref = cv2.imread("photo.jpg")
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = ref[1800:2125, 1410:1750]
#plt.imshow(ref)
#plt.show()

#1. Noise Reduction by blurring the image
#find g = conv(f,h), h is gaussian filter
#gaussian filter formula:
# (2* pi * sigma**2)^-1  * e ^ (  -( i - (k + 1)**2 + j - (k + 1)**2) / 2 * sigma**2 )
def gaussian_filter(image, kernel_size, sigma):
    k = (kernel_size - 1)// 2
    #we creating these to matrices:
    '''
    x : 
    [[-2. -1.  0.  1.  2.]
    [-2. -1.  0.  1.  2.]
    [-2. -1.  0.  1.  2.]
    [-2. -1.  0.  1.  2.]
    [-2. -1.  0.  1.  2.]]

    y : 
    [[-2. -2. -2. -2. -2.]
    [-1. -1. -1. -1. -1.]
    [ 0.  0.  0.  0.  0.]
    [ 1.  1.  1.  1.  1.]
    [ 2.  2.  2.  2.  2.]]
    '''
    x , y = np.meshgrid(np.linspace(-k, k , kernel_size), np.linspace(-k, k, kernel_size))
    gaussian_kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2)/(2 * sigma**2))
    #normalise so that the image wont darken:
    gaussian_kernel /= np.sum(gaussian_kernel) 
    g = cv2.filter2D(image, -1, gaussian_kernel)
    return g

#as sigma increases, blur increases
noise_reduced_image = gaussian_filter(ref, 5, 5)
#plt.imshow(noise_reduced_image, cmap='gray')
#plt.show()



def compute_gradient(image):
    #Sobel filter:
    Kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], np.float32)
    Kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], np.float32)
    derivative_x = cv2.filter2D(image, cv2.CV_64F, Kernel_x)
    derivative_y = cv2.filter2D(image, cv2.CV_64F, Kernel_y)

    magnitude = np.sqrt(derivative_x**2 + derivative_y**2)
    theta = np.arctan2(derivative_y, derivative_x)

    return magnitude, theta



   # Derivative_x = cv2.filter2D(image, -1, Kernel_x)
   # Derivative_y = cv2.filter2D(image, -1, Kernel_y)

   # magnitude = np.sqrt(Derivative_x**2 + Derivative_y**2)
    '''
    derivative_x = cv2.filter2D(image, -1, Kernel_x)
    derivative_y = cv2.filter2D(image, -1, Kernel_y)

    magnitude = np.sqrt(derivative_x**2 + derivative_y**2)
    theta = np.arctan2(derivative_y, derivative_x)
    magnitude = magnitude / magnitude.max() * 255
    Derivative_x = ndimage.convolve(image, Kernel_x)
    Derivative_y = ndimage.convolve(image, Kernel_y)
    magnitude = np.hypot(Derivative_x, Derivative_y)
    magnitude = magnitude / magnitude.max() * 255
    theta = np.arctan2(Derivative_y, Derivative_x)
    '''




def non_max_suppression(magnitude, angle):
    suppressed = np.zeros_like(magnitude)
    angle = angle * (180.0 / np.pi)
    angle[angle < 0] += 180

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed[i, j] = magnitude[i, j]

    return suppressed



mag, angle = compute_gradient(noise_reduced_image)
cv2.imshow('Original Image', noise_reduced_image)
cv2.imshow('Gradient Magnitude', mag.astype(np.uint8))
cv2.imshow('Gradient Direction', angle)
suppr = non_max_suppression(mag, angle)
cv2.imshow('Gradient Suppressed', suppr)
cv2.waitKey(0)
cv2.destroyAllWindows()
