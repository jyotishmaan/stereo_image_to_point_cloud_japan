'''
Opening a video file under path IN_FILE, assuming that the video contain stereo images
Cutting N frames and storing stereo images with range
((L_MIN_X,  L_MAX_X), (L_MIN_Y,L_MAX_Y)), (R_MIN_X,  R_MAX_X), (R_MIN_Y,R_MAX_Y))
relative to the video
under output path OUT_FILE_L and OUT_FILE_R using the name convention
- frame_#_left.jpg
- frame_#_right.jpg,
respectively

If N <= 0, then it will process the entire video
'''
import os
import cv2
IN_FILE = "../data/raw_data/both_eye_0304_1.mp4"
OUT_FILE_L = "../data/video_frames/left/"
OUT_FILE_R = "../data/video_frames/right/"
N = 100
L_MIN_X = 60
L_MAX_X = 1260
L_MIN_Y = 160
L_MAX_Y = 910

R_MIN_X = 1340
R_MAX_X = 2540
R_MIN_Y = 160
R_MAX_Y = 910

if not os.path.exists(OUT_FILE_L):
    print("Creating " + OUT_FILE_L + " because it does not exist")
    os.makedirs(OUT_FILE_L)

if not os.path.exists(OUT_FILE_R):
    print("Creating " + OUT_FILE_R + " because it does not exist")
    os.makedirs(OUT_FILE_R)

if(L_MAX_X - L_MIN_X != R_MAX_X - R_MIN_X or L_MAX_Y - L_MIN_Y != R_MAX_Y - R_MIN_Y):
    print("ALERT: Left and Right Stereo Image size does not match")

vidcap = cv2.VideoCapture(IN_FILE)
success, image = vidcap.read()
count = 0
success = True
# image[y,x]
imageL = image[L_MIN_Y:L_MAX_Y:, L_MIN_X:L_MAX_X]
imageR = image[R_MIN_Y:R_MAX_Y, R_MIN_X:R_MAX_X]

while success and count < N:
    cv2.imwrite(OUT_FILE_L + "frame_{}_left.jpg".format(count), imageL)
    cv2.imwrite(OUT_FILE_R + "frame_{}_right.jpg".format(count), imageR)
    success, image = vidcap.read()
    print('Frame # {}  status: {}'.format(count, "OK" if success else "ERROR"))
    count += 1
