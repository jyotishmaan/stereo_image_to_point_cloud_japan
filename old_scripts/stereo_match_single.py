#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

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
    print('loading images...')
    imgL = cv2.imread(
        "./data/raw_data/left.png")
    imgR = cv2.imread(
        "./data/raw_data/right.png")

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

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey()

    # print('generating 3d point cloud...',)
    # h, w = imgL.shape[:2]
    # f = 0.8*w                          # guess for focal length
    # Q = np.float32([[1, 0, 0, -0.5*w],
    #                 [0, -1, 0,  0.5*h],  # turn points 180 deg around x-axis,
    #                 [0, 0, 0,     -f],  # so that y-axis looks up
    #                 [0, 0, 1,      0]])
    # points = cv2.reprojectImageTo3D(disp, Q)
    # colors = cv2.cv2tColor(imgL, cv2.COLOR_BGR2RGB)
    # mask = disp > disp.min()
    # out_points = points[mask]
    # out_colors = colors[mask]
    # out_fn = "stereo_match.ply"
    # write_ply(out_fn, out_points, out_colors)
    # print('%s saved' % 'out.ply')

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
