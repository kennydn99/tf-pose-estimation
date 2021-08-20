import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys
import argparse
import cv2
import time
import os
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from deeplifting.packages.lifting.utils.prob_model import Prob3dPose

class Grid(object):
    def __init__(self):
        #create window and Initialize graph objects
        self.app = QtGui.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Grid')
        self.window.setGeometry(0, 110, 1920, 1080)
        self.window.setCameraPosition(distance=30, elevation=12)
        self.window.show()

        x = gl.GLGridItem()
        y = gl.GLGridItem()
        z = gl.GLGridItem()
        x.rotate(90, 0, 1, 0)
        y.rotate(90, 1, 0, 0)
        x.translate(-10, 0, 0)
        y.translate(0, -10, 0)
        z.translate(0, 0, -10)
        self.window.addItem(x)
        self.window.addItem(y)
        self.window.addItem(z)
        
        #create model
        model = 'cmu'

        #dictionary to hold all line segments
        self.lines = {}
        self.connection = [
            [0, 1], [1, 2], [2, 3], [0, 4],
            [4, 5], [5, 6], [0, 7], [7, 8],
            [8, 9], [9, 10], [8, 11], [11, 12],
            [12, 13], [8, 14], [14, 15], [15, 16]
        ]

        self.w, self.h = model_wh(args.resize)
        #432x368 default target size
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(self.w, self.h))
        #open and read video
        self.cam = cv2.VideoCapture(args.video)
        ret_val, image = self.cam.read()
        #load mat file for 3d pose estimation
        self.poseLifting = Prob3dPose('./deeplifting/data/saved_sessions/prob_model/prob_model_params.mat')
        keypoints = self.mesh(image)

        #create scatterplot object
        self.points = gl.GLScatterPlotItem(
            pos=keypoints,
            color=pg.glColor((51,255,153)), 
            size=15
        )
        self.window.addItem(self.points)
        #create connecting lines and plot
        for idx, pts in enumerate(self.connection):
            self.lines[idx] = gl.GLLinePlotItem(
                pos = np.array([keypoints[p] for p in pts]),
                color = pg.glColor((153,51,255)),
                width=3,
                antialias = True
            )
            self.window.addItem(self.lines[idx])

    def mesh(self, image):
        #return 3d keypoints, do inference based on image
        image_h, image_w = image.shape[:2]
        width = 640
        height = 480
        
        pose_2d_mpiis = []
        visibilities = []
        
        humans = self.e.inference(
            image,
            resize_to_default=(self.w > 0 and self.h > 0),
            upsample_size=args.resize_out_ratio,
        )
        
        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            #get 2d keypoints
            pose_2d_mpiis.append([(int(x * width + 0.5), int(y * height + 0.5)) for x,y in pose_2d_mpii])
            visibilities.append(visibility)
        
        pose_2d_mpiis = np.array(pose_2d_mpiis)
        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
        #get 3d points
        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)

        #[x, y, z] format
        keypoints = pose_3d[0].transpose()
        for point in keypoints:
            print(point)
        return keypoints / 80

    def update(self):
        #update all graph objects

        ret_val, image = self.cam.read()
        try:
            #get 3d keypoints
            keypoints = self.mesh(image)
        except AssertionError:
            print('no humans found in image')
        else:
            #update scatterplot
            self.points.setData(pos=keypoints)
            #update lines
            for idx, pts in enumerate(self.connection):
                self.lines[idx].setData(
                    pos = np.array([keypoints[p] for p in pts])
                )

    def start(self):
        #open grid window and setup
        if(sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
    
    def animation(self, frametime=10):
        # call update and lopp it
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(frametime)
        self.start()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="3dvideo")
    
    parser.add_argument("--video", type=str, default='0')
    parser.add_argument(
        "--resize",
        type=str,
        default="0x0",
        help="if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ",
    )
    parser.add_argument(
        "--resize-out-ratio",
        type=float,
        default=4.0,
        help="if provided, resize heatmaps before they are post-processed. default=1.0",
    )
    args = parser.parse_args()
    
    g = Grid()
    g.animation()