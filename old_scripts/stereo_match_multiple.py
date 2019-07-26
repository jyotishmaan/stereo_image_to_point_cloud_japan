# Python 2/3 compatibility

#!/usr/bin/env python
"""
Adapted from example of stereo image matching and point cloud generation tutorial from OpenCV
https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py

INFILE_L -- directory where the input left images are located
INFILE_R -- directory where the input right images are located

OUTFILE_PT -- point cloud output directory with format frame_{}_pointcloud.ply
OUTFILE_DISPARITY -- disparity map output directrory with format -- frame_{}_disaprity.jpg

N -- number of stereo images / frames

**kargs -- arguments inherited for StereoSGBM_create function, please see default values below
"""
from __future__ import print_function

import numpy as np
import cv2
import os


INFILE_L = "./data/video_frames/left"
INFILE_R = "./data/video_frames/right"

OUTFILE_PT = "./data/point_cloud_output"
OUTFILE_DISPARITY = "./data/disparity_output"

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

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    if not os.path.exists(OUTFILE_PT):
        print("Creating " + OUTFILE_PT + " because it does not exist")
        os.makedirs(OUTFILE_PT)

    if not os.path.exists(OUTFILE_DISPARITY):
        print("Creating " + OUTFILE_DISPARITY + " because it does not exist")
        os.makedirs(OUTFILE_DISPARITY)

    ## CHECK IF THERE ARE EQUAL NUMBER OF LEFT AND RIGHT IMAGES ##
    left_image_paths = [os.path.join(INFILE_L, n) for n in os.listdir(
        INFILE_L) if n[len(n) - 4:] == ".jpg"]
    right_image_paths = [os.path.join(INFILE_R, n) for n in os.listdir(
        INFILE_R) if n[len(n) - 4:] == ".jpg"]
    assert len(left_image_paths) == len(
        right_image_paths), "ERROR: The number of left and right images are different"
    assert len(left_image_paths) != 0, "ERROR: There are no images to be read"

    # create stereoSGBM mask
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

    counter = 0
    while counter < N and counter < 1:
        # Read in left and right image
        imgL = cv2.imread(os.path.join(
            INFILE_L, "frame_{}_left.jpg".format(counter)))
        imgR = cv2.imread(os.path.join(
            INFILE_R, "frame_{}_right.jpg".format(counter)))

        # compute disparity
        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        # print(disp[100:150,120:160])
        displayed_image = (disp-min_disp)/num_disp
        # cv2.rectangle(displayed_image, (100, 120), (150 , 160),
        #                       (255, 255, 255), 2)
        # cv2.imshow('disparity', diplayed_image) # for some reason when using imshow, the picture has to be normalized
        # cv2.waitKey()
        # generate 3D point cloud
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
        cv2.imwrite(os.path.join(OUTFILE_DISPARITY, "frame_{}_disparity.jpg".format(
            counter)), (disp))  # but when saving, normalization is not necessary
        write_ply(os.path.join(OUTFILE_PT, "frame_{}_pointcloud.ply".format(
            counter)), out_points, out_colors)
        print("Frame {} processed, all data saved".format(counter))
        counter += 1


if __name__ == '__main__':
    # print(__doc__)
    main()
    cv2.destroyAllWindows()
