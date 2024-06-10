Part 1: GUI Application

Import necessary libraries:

* tkinter for creating the GUI
* filedialog from Tkinter for opening file dialogs
* PIL (Python Imaging Library) for image handling
* matplotlib.pyplot for displaying images
* cv_depth_blur (a custom module) for the depth-based blur function


Initialize an empty list image_paths to store the paths of selected images.
Define functions:

* run_code(param1): Calls get_blur_image with two image paths and a subject index, then displays the result using matplotlib.
* open_image(): Opens a file dialog to select an image and appends its path to image_paths.
* run_with_parameters(): Gets the subject index from the entry field and calls run_code.


Create the main Tkinter window:

* Set the title, size, and position the window at the center of the screen.
* Set the background color to dark green.


Create GUI elements:

* Buttons to open two images and run the code.
* An entry field and label for the subject index parameter.


Start the Tkinter main event loop.




Part 2: Depth-Based Background Blur (cv_depth_blur.py)

Import libraries:

* cv2 (OpenCV) for image processing and computer vision tasks.
* numpy for numerical operations.
* skimage.filters for multi-Otsu thresholding.


Define the get_blur_image function that takes two image paths and a subject index:
a. Load images:

* imgo: Original color image (RGB).
* img1 and img2: Grayscale images for stereo matching.

b. Feature matching using SIFT and FLANN:

* SIFT (Scale-Invariant Feature Transform): A feature detection algorithm that finds keypoints and descriptors in images. It's invariant to scale, rotation, and partially invariant to illumination changes.
* FLANN (Fast Library for Approximate Nearest Neighbors): Used for fast matching of SIFT descriptors between the two images.
Lowe's ratio test is used to filter good matches.

c. Fundamental Matrix estimation:

Using cv.findFundamentalMat with LMEDS (Least Median of Squares) method to estimate the fundamental matrix F.
The fundamental matrix encodes the epipolar geometry between two views of the same scene.

d. Stereo Rectification:

* cv.stereoRectifyUncalibrated is used to compute rectification transforms H1 and H2.
Rectification aligns the epipolar lines horizontally, making stereo matching easier.
* cv.warpPerspective applies these transforms to rectify the images.

e. Stereo Matching:

cv.StereoSGBM_create (Semi-Global Block Matching) is used to compute the disparity map.
Disparity is the difference in x-coordinates of corresponding points in stereo images. It's inversely proportional to depth.

f. Depth Segmentation:

Multi-Otsu thresholding (from skimage.filters) is used to segment the disparity map into 4 regions.
This effectively segments the image into 4 depth ranges.

g. Selective Blurring:

For each depth region (except the subject), apply Gaussian blur using cv.GaussianBlur.

The blur amount (blur_dict) and region order (idx_dict) are predefined based on the subject index.

The blurred regions are then combined with the original image to create the final output.



Computer Vision Concepts:

Feature Detection and Matching (SIFT):

Keypoints are distinctive points in an image (corners, blobs, etc.).

Descriptors are vectors that describe the neighborhood around keypoints.

Matching finds correspondences between keypoints in different images.


Epipolar Geometry:

Describes the geometry between two views of the same 3D scene.

The fundamental matrix F encapsulates this geometry.

Epipolar lines are lines in one image where the corresponding point in the other image must lie.


Stereo Rectification:

Transforms stereo images so that corresponding points lie on the same horizontal scanline.

Makes stereo matching a 1D search problem instead of 2D, improving efficiency and accuracy.


Stereo Matching and Disparity:

Finds corresponding points between stereo images.

Disparity is higher for closer objects and lower for distant ones, giving depth information.

SGBM is a robust method that considers both local (block matching) and global (energy minimization) constraints.


Image Segmentation:

Partitioning an image into multiple segments or regions.

Here, multi-Otsu thresholding segments the disparity map based on depth.


Image Filtering:

Gaussian blur is a linear filter that reduces image noise and detail.

The kernel size ((5, 5), (9, 9), etc.) affects the blur strength.



This code demonstrates a creative application of stereo vision for depth estimation and using that information for selective background blurring, similar to the "portrait mode" in modern smartphones.
