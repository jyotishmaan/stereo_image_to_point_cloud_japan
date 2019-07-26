import os
import cv2
IN_FILE = "../data/raw_data/both_eye_0304_1.mp4"
OUT_FILE = "../data/single_camera_left/single_left_cam_output.mp4"
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
    frame = frame[L_MIN_Y:L_MAX_Y:, L_MIN_X:L_MAX_X]
    if ret==True:
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()
