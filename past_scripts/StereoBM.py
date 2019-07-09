
### CANNOT USE SINCE StereoBM_create requires a rectified image, but image rectification can be only done with camera calibration
import numpy
import cv2


from matplotlib import pyplot as plt
from matplotlib import cm
import pyviz3d as viz

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
    verts = numpy.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        numpy.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


left = cv2.imread(
    "/Users/michaelwu/Desktop/projects/japan/data/left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread(
    "/Users/michaelwu/Desktop/projects/japan/data/right.png", cv2.IMREAD_GRAYSCALE)
h, w = left.shape[:2]
fx =  0.8*w                          # guess for focal length       # lense focal length
baseline = 200 #30   # distance in mm between the two cameras
disparities = 32 #128  # num of disparities to consider
block = 17#31        # block size to match
units = 5     # depth units, adjusted for the output to fit in one byte

sbm = cv2.StereoBM_create(numDisparities=disparities,
                          blockSize=block)

# calculate disparities
disparity = sbm.compute(left, right)
valid_pixels = disparity > 0

# calculate depth data
depth = numpy.zeros(shape=left.shape).astype("uint8")
depth[valid_pixels] = (fx * baseline) / (units * disparity[valid_pixels])

# visualize depth data
depth = cv2.equalizeHist(depth)
colorized_depth = numpy.zeros((left.shape[0], left.shape[1], 3), dtype="uint8")
temp = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
colorized_depth[valid_pixels] = temp[valid_pixels]
# plt.imshow(depth)
plt.imshow(colorized_depth)
plt.show()
#
# Q = numpy.float32([[1, 0, 0, -0.5*w],
#                 [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
#                 [0, 0, 0,     -fx], # so that y-axis looks up
#                 [0, 0, 1,      0]])
# points = cv2.reprojectImageTo3D(disparity,Q)
# colors = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
# mask = disparity > -1000000
# out_points = points[mask]
# out_colors = colors[mask]
# out_fn = 'script_out.ply'
# write_ply(out_fn, out_points, out_colors)
# viz.show_pointclouds([out_points], [out_colors])  # Display point cloud
