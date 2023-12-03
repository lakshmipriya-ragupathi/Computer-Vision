import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

def get_blur_image(img1_path, img2_path, subject_index):
    # Parameters
    imgo = cv.imread(img1_path)
    imgo = cv.cvtColor(imgo, cv.COLOR_BGR2RGB)

    img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)  # left image
    img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)  # right image

    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    ############## Stereo rectify uncalibrated ##############
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    thresh = 0
    _, H1, H2 = cv.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1), threshold=thresh,
    )

    ############## Undistort (Rectify) ##############
    imgL_undistorted = cv.warpPerspective(img1, H1, (w1, h1))
    imgR_undistorted = cv.warpPerspective(img2, H2, (w2, h2))

    win_size = 2
    min_disp = -7
    max_disp = 9
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=5,
        uniquenessRatio=5,
        speckleWindowSize=5,
        speckleRange=5,
        disp12MaxDiff=2,
        P1=8 * 3 * win_size ** 2,
        P2=32 * 3 * win_size ** 2,
    )
    disparity_SGBM = stereo.compute(imgL_undistorted, imgR_undistorted)

    image = disparity_SGBM
    thresholds = threshold_multiotsu(image, classes=4)

    # Using the threshold values, we generate the 4 regions.
    regions = np.digitize(image, bins=thresholds)
    masks = [np.zeros(shape=image.shape) for i in range(4)]

    for i in range(4):
        masks[i][regions == i] = 1

    out = imgo

    blur_dict = {0: [5, 9, 15], 1: [5, 9, 13], 2: [15, 9, 25], 3:[25, 15, 9]}
    idx_dict = {0: [1, 2, 3], 1: [0, 2, 3], 2:[0, 1, 3], 3:[0, 1, 2]}

    j = 0
    #blurring regions
    for i in idx_dict[subject_index]:
        blur = cv.GaussianBlur(imgo, (blur_dict[subject_index][j], blur_dict[subject_index][j]), 0)
        j += 1
        out[regions == i] = blur[regions == i]
    
    return out
