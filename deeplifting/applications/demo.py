#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

import __init__
import csv
from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os.path import dirname, realpath

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/abduction_90.jpg'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'


def main():
    image = cv2.imread(IMAGE_FILE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

    # create pose estimator
    image_size = image.shape

    pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    try:
        # estimation
        pose_2d, visibility, pose_3d = pose_estimator.estimate(image)
        
        # print out transposed 3d keypoints
        pose_3dqt = np.array(pose_3d[0].transpose())
        for p in pose_3dqt:
            print(p)
        
        # Try to put 3d keypoints in csv file
        writeToCSV('3dkeypoints.csv', pose_3dqt)

        # Show 2D and 3D poses
        display_results(image, pose_2d, visibility, pose_3d)
    except ValueError:
        print('No visible people in the image. Change CENTER_TR in packages/lifting/utils/config.py ...')

    # close model
    pose_estimator.close()

def writeToCSV(filename, pose_data3d):
    """write 3d keypoints data to a csv file"""
    header = ['Bodypart', 'X', 'Y', 'Z']
    bodypart_name = [
        "Bottom torso", "HipRight", "KneeRight", "FootRight",
        "HipLeft", "KneeLeft", "FootLeft", "SpineMid",
        "SpineShoulder", "NeckBase", "CenterHead", "ShoulderLeft",
        "ElbowLeft", "WristLeft", "ShoulderRight", "ElbowRight", "WristRight"
    ]
    idx = 0
    rows = []
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for data in pose_data3d:
            rows.append([bodypart_name[idx], "{:.2f}".format(data[0]), "{:.2f}".format(data[1]), "{:.2f}".format(data[2])])
            idx += 1
        writer.writerows(rows)


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())
