import numpy as np
from sklearn.preprocessing import normalize
import cv2

IN_FILE = "./data/raw_data/both_eye_0304_1.mp4"
OUT_FILE = "./data/new_output.mp4"
L_MIN_X = 60
L_MAX_X = 1260
L_MIN_Y = 160
L_MAX_Y = 910

R_MIN_X = 1340
R_MAX_X = 2540
R_MIN_Y = 160
R_MAX_Y = 910


cap = cv2.VideoCapture(IN_FILE)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUT_FILE,fourcc, 20.0, (1200,750))

while(cap.isOpened()):
    ret, frame = cap.read()
    imgL = frame[L_MIN_Y:L_MAX_Y:, L_MIN_X:L_MAX_X]
    imgR = frame[R_MIN_Y:R_MAX_Y, R_MIN_X:R_MAX_X]
    if ret == True:
        # SGBM Parameters -----------------
        window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=5,
            P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
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

        print('computing disparity...')
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        cv2.imshow('Disparity Map', filteredImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
