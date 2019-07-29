

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

        N = 100

        window_size = 2
        min_disp = 16
        num_disp = 192-min_disp*2
        blockSize = window_size
        uniquenessRatio = 1
        speckleRange = 50
        speckleWindowSize = 200
        disp12MaxDiff = 200
        P1 = 600
        P2 = 2400
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
        )
        # compute disparity
        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        # print(disp[100:150,120:160])
        displayed_image = (disp-min_disp)/num_disp
        cv2.imshow('disparity', displayed_image) # for some reason when using imshow, the picture has to be normalized
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
