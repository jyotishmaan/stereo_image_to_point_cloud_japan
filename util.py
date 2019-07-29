import numpy as np
from sklearn.preprocessing import normalize
import cv2

def calculate_disparity_michael(imgL,
                                imgR,
                                window_size=2,
                                min_disp=16,
                                num_disp=160,
                                blockSize=2,
                                uniquenessRatio=1,
                                speckleRange=50,
                                speckleWindowSize=200,
                                disp12MaxDiff=200,
                                P1=600,
                                P2=2400,
                                preFilterCap=63,
                                show_disparity=False,
                                save_point_cloud=False):
    stereo = cv2.StereoSGBM_create(
        # Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
        minDisparity=min_disp,
        # Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
        numDisparities=num_disp,
        blockSize=window_size,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1,
        P2=P2,
        preFilterCap=preFilterCap
    )
    # compute disparity
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    if show_disparity:
        displayed_image = (disp-min_disp)/num_disp
        cv2.imshow('disparity', displayed_image) # for some reason when using imshow, the picture has to be normalized
    if save_point_cloud:
        h, w = imgL.shape[:2]
        f = 0.8*w                          # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5*w],
                        [0, -1, 0,  0.5*h],  # turn points 180 deg around x-axis,
                        [0, 0, 0,     -f],  # so that y-axis looks up
                        [0, 0, 1,      0]])
        points = cv2.reprojectImageTo3D(disp, Q)
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        mask = disp > disp.min()
        out_points = points[mask]
        out_colors = colors[mask]
        # saving files
        cv2.imwrite(os.path.join("./data/disparity", "frame_{}_disparity.jpg".format(
            counter)), (disp))  # but when saving, normalization is not necessary
        write_ply(os.path.join("./data/point_cloud", "frame_{}_pointcloud.ply".format(
            counter)), out_points, out_colors)
    return disp

def calculate_disparity_tim(imgL,
                            imgR,
                            window_size=3,
                            min_disp=0,
                            num_disp=160,
                            blockSize=5,
                            uniquenessRatio=15,
                            speckleRange=2,
                            speckleWindowSize=0,
                            disp12MaxDiff=1,
                            P1=216,
                            P2=864,
                            preFilterCap=63,
                            show_disparity=False,
                            save_point_cloud=False):
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=blockSize,
        P1=P1,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=P2,
        disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        preFilterCap=preFilterCap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    if show_disparity:
        cv2.imshow('disparity', filteredImg) # for some reason w
    return filteredImg
