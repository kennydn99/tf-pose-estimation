'''Must delete CSV folder to run again'''

import argparse
import logging
import time
import math
import cv2
import numpy as np
import csv
import os

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from deeplifting.packages.lifting.utils.prob_model import Prob3dPose
from deeplifting.applications.demo import createCSV, addDataToCSV
from deeplifting.packages.lifting._pose_estimator import PoseEstimator

logger = logging.getLogger("TfPoseEstimator-WebCam")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tf-pose-estimation realtime webcam")
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

    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_thin",
        help="cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small",
    )
    parser.add_argument(
        "--show-process",
        type=bool,
        default=False,
        help="for debug purpose, if enabled, speed for inference is dropped.",
    )

    parser.add_argument(
        "--tensorrt", type=str, default="False", help="for tensorrt process."
    )
    args = parser.parse_args()

    logger.debug("initialization %s : %s" % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(
            get_graph_path(args.model),
            target_size=(w, h),
            trt_bool=str2bool(args.tensorrt),
        )
    else:
        e = TfPoseEstimator(
            get_graph_path(args.model),
            target_size=(432, 368),
            trt_bool=str2bool(args.tensorrt),
        )
    logger.debug("cam read+")
    cam = cv2.VideoCapture(args.video)
    #ret_val, image = cam.read()
    #logger.info("cam image=%dx%d" % (image.shape[1], image.shape[0]))

    #funtion to find midpoint coordinates between two points
    def midpoint(first, last):
        first = np.array(first)
        last = np.array(last)
        return [(first[0]+last[0])/2, (first[1]+last[1])/2]

    #function to create csv files into csv directory
    def createCSV(filename):
        header = ["Frame Number", 'X', 'Y', 'Z']
        with open("./CSV/" + filename + '.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(header)

    #function to add data into existing csv file
    def addDataToCSV(file, newdata):
        with open("./CSV/" + file + '.csv', 'a') as f:
            w = csv.writer(f)
            w.writerow(newdata)

    #variables for getting 3d data
    poseLifting = Prob3dPose('./deeplifting/data/saved_sessions/prob_model/prob_model_params.mat')
    
    #image_h, image_w = image.shape[:2]
    default_w = 640
    default_h = 480
    pose_2d_mpiis = []
    visibilities = []

    #variables for writing to csv file
    frameNum = 1
    bodypart_dict = {
        0:"Bottom torso", 1:"HipRight", 2:"KneeRight", 3:"FootRight",
        4:"HipLeft", 5:"KneeLeft", 6:"FootLeft", 7:"SpineMid",
        8:"SpineShoulder", 9:"NeckBase", 10:"CenterHead", 11:"ShoulderLeft",
        12:"ElbowLeft", 13:"WristLeft", 14:"ShoulderRight", 15:"ElbowRight", 16:"WristRight"
    }
    data3dpoints = []
    
    #make folder to store all csv files
    try:
        os.mkdir("./CSV")
    except OSError as e:
        print("The directory exists")
    #first create csv files for all body parts
    for val in bodypart_dict.values():
        createCSV(val)
    
    #find total num of frames
    totalFrameNum = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = totalFrameNum/1
    fno = 0
    success, image = cam.read()
    
    #create other pose estimator for 3d
    pose_estimator = PoseEstimator(
            image.shape,
            './deeplifting/data/saved_sessions/init_session/init',
            './deeplifting/data/saved_sessions/prob_model/prob_model_params.mat'
        )
    pose_estimator.initialise()

    while success:
        if fno % sample_rate == 0:
            ret_val, image = cam.retrieve()
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            logger.debug("image process+")
            
            humans = e.inference(
                image,
                resize_to_default=(w > 0 and h > 0),
                upsample_size=args.resize_out_ratio,
            )

            logger.debug("postprocess+")
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            
            try:
                for human in humans:
                    #save coordinates of shoulder, hip, and elbow & neck
                    neck = [human.body_parts[1].x, human.body_parts[1].y]
                    RH = [human.body_parts[8].x, human.body_parts[8].y]
                    LH = [human.body_parts[11].x, human.body_parts[11].y]
                
                img_h, img_w = image.shape[:2]
                #draw spine and visualize center of gravity
                mid = midpoint(RH, LH)
                cg = midpoint(neck, mid)
                cv2.line(
                        image,
                        tuple(np.multiply(neck, [img_w, img_h]).astype(int)),
                        tuple(np.multiply(mid, [img_w, img_h]).astype(int)),
                        (255,0,0),
                        3
                    )
                cv2.circle(
                        image,
                        tuple(np.multiply(cg, [img_w, img_h]).astype(int)),
                        3,
                        (0,255,0),
                        thickness=3,
                        lineType=8,
                        shift=0,
                    )
            except:
                pass

            logger.debug("show+")
            cv2.putText(
                image,
                "FPS: %f" % (1.0 / (time.time() - fps_time)),
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            cv2.imshow("tf-pose-estimation result", image)
            fps_time = time.time()
            
            
            # collect 3d data points
            for human in humans:
                #collect 2d keypoints
                pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
                pose_2d_mpiis.append([(int(x * default_w + 0.5), int(y * default_h + 0.5)) for x,y in pose_2d_mpii])
                visibilities.append(visibility)

            # collect 3d points for current frame
            pose_2d_mpiis = np.array(pose_2d_mpiis)
            visibilities = np.array(visibilities)
            transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
            pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
            transposed_3dpoints = np.array(pose_3d[0]).transpose()
            #change nparray back to list
            pose_2d_mpiis = pose_2d_mpiis.tolist()
            visibilities = visibilities.tolist()
            '''
            for human in humans:
                pose_2d_mpii, visibility, newpose3d = pose_estimator.estimate(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
            transposed_3dpoints = np.array(newpose3d[0]).transpose()'''

            #add transposed 3d data and current frame to 3d data list and update csv file
            for data in transposed_3dpoints:
                data3dpoints.append([frameNum, "{:.2f}".format(data[0]), "{:.2f}".format(data[1]), "{:.2f}".format(data[2])])
            for index, bodypart in bodypart_dict.items():
                addDataToCSV(bodypart, data3dpoints[index])

            #clear lists
            pose_2d_mpiis.clear()
            visibilities.clear()
            data3dpoints.clear()
            frameNum += 1
        #next frame
        success, image = cam.read()

        if cv2.waitKey(1) == 27:
            break
        logger.debug("finished+")
    
    cam.release()
    cv2.destroyAllWindows()