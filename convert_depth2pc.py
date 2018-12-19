import argparse
import sys
import os
from PIL import Image

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# focalLength = 938.0
# centerX = 319.5
# centerY = 239.5
# scalingFactor = 1

# # for kinect (RGB):
# # distortion coefficients: 0.2402, -0.6861, -0.0015, 0.0003
# focalLength = 524 # in mm = 2.9
# centerX = 316.7
# centerY = 238.5
# scalingFactor = 1

# for kinect (IR):
# distortion coefficients: -0.1296, 0.45, -0.0005, -0.0002
# focalLength = 585.6 # in mm = 6.1
focalLength = 938.0 # in mm = 6.1
centerX = 316
centerY = 247.6
scalingFactor = 1

def generate_pointcloud(depth_file,ply_file):
    depth = Image.fromarray(depth_file, 'I')
    # w, h = np.shape(depth_file)
    
    # print("array shape: ", np.shape(depth_file))
    
    # depth = Image.open(depth_file).convert('I')

    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")


    points = []    
    for v in range(depth.size[1]):
        for u in range(depth.size[0]):
            Z = depth.getpixel((u,v)) / scalingFactor
            # print(Z)
            if Z==0: continue
            # X = (u - centerX) * Z / focalLength
            # Y = (v - centerY) * Z / focalLength
            X = (u)# - centerX) * Z / focalLength
            Y = (v)# - centerY) * Z / focalLength
            points.append((X,Y,Z))
            # points.append("%f %f %f 0\n"%(X,Y,Z))

    # print(np.max(points))
    print("number of elements: ", np.shape(points))
    show_sample(np.array(points))

    # visualize the point cloud 

def show_sample(x):
    # create plot object 
    print("shape of x: ", np.shape(x))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 270)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_zticks([])

    # fill the plot with data (auto)
    # for i, (c, m) in enumerate([('r', 'o'), ('b', '^'), ('y', 'X'), ('g', 'v')]):
    #     xs = x[:,i][:,0]
    #     ys = x[:,i][:,1]
    #     zs = x[:,i][:,2]
    #     ax.scatter(xs, ys, zs, s=1.5, c=c, marker=m)
    xs = x[:,0]
    ys = x[:,1]
    zs = x[:,2]
    ax.scatter(xs, zs, ys, s=1.5, c='b', marker='x')
    plt.show()

# def generate_pointcloud_original(rgb_file,depth_file,ply_file):

#     rgb = Image.open(rgb_file)
#     depth = Image.open(depth_file).convert('I')

#     if rgb.size != depth.size:
#         raise Exception("Color and depth image do not have the same resolution.")
#     if rgb.mode != "RGB":
#         raise Exception("Color image is not in RGB format")
#     if depth.mode != "I":
#         raise Exception("Depth image is not in intensity format")


#     points = []    
#     for v in range(rgb.size[1]):
#         for u in range(rgb.size[0]):
#             color = rgb.getpixel((u,v))
#             Z = depth.getpixel((u,v)) / scalingFactor
#             print(Z)
#             if Z==0: continue
#             X = (u - centerX) * Z / focalLength
#             Y = (v - centerY) * Z / focalLength
#             points.append("%f %f %f %d %d %d 0\n"%)