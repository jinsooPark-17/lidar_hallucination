#!/usr/bin/env python3
import numpy as np
from skimage import filters

import rospy
from sensor_msgs.msg import LaserScan

rospy.init_node('test_scan2img')
rospy.sleep(1.0)

scan = rospy.wait_for_message("roberto/scan_filtered", LaserScan, timeout=10.0)

r     = scan.ranges
valid = np.isfinite(r)
theta = np.arange(scan.angle_min, scan.angle_max + scan.angle_increment, scan.angle_increment)

Xs = (r * np.cos(theta) / 0.1)[valid].astype(np.int)
Ys = (r * np.sin(theta) / 0.1)[valid].astype(np.int)

scan_img = np.zeros((128, 128), dtype=np.uint8)
scan_img[Xs, Ys] = 1.0


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for inflation in range(1,21,2):
    plt.figure()
    blurred = filters.gaussian( scan_img, sigma=(1.0, 1.0), truncate = inflation/2)
    print(blurred.shape)
    print(np.unique(blurred))
    plt.imsave(f"inflation_{inflation}px.png", blurred, dpi=300)
