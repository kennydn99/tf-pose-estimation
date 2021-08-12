import argparse
import logging
import sys
import time

import math

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from deeplifting.packages.lifting.utils.prob_model import Prob3dPose

logger = logging.getLogger("TfPoseEstimatorRun")
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tf-pose-estimation run")
    parser.add_argument("--image", type=str, default="./images/p1.jpg")
    parser.add_argument(
        "--model",
        type=str,
        default="cmu",
        help="cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="0x0",
        help="if provided, resize images before they are processed. "
        "default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ",
    )
    parser.add_argument(
        "--resize-out-ratio",
        type=float,
        default=4.0,
        help="if provided, resize heatmaps before they are post-processed. default=1.0",
    )

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    if image is None:
        logger.error("Image can not be read, path=%s" % args.image)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(
        image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio
    )
    elapsed = time.time() - t

    logger.info("inference image: %s in %.4f seconds." % (args.image, elapsed))

    #calculate angle function given three joints
    def calculate_angle(x,y,z):
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        #calculate angle between three coordinates
        rad = np.arctan2(z[1]-y[1], z[0]-y[0]) - np.arctan2(x[1]-y[1], x[0]-y[0])
        angle = np.abs(180 * rad / np.pi)
        #max angle = 180 degrees
        if angle > 180.0:
            angle = 360 - angle
        return angle

    #find perpendicular coordinates function - pass in two joint coordinates 
    def perpCoord(a,b):
        a = np.array(a)
        b = np.array(b)
        # find length between a and b
        length = np.linalg.norm(a-b)
        #get direction vector = [delta x, delta y]
        dv = [b[0] - a[0], b[1] - a[1]]
        if dv[0] == 0 or dv[1] == 0:
            c = [b[0], b[1] + length]
            return c, dv
        #get magnitude
        var = math.sqrt(dv[0]*dv[0] + dv[1]*dv[1])
        dv[0] = dv[0]/var
        dv[1] = dv[1]/var
        #invert direction vector coordinate and swap
        dv[0], dv[1] = -dv[1], dv[0]
        # new line starting at b pointing in direction of dv
        if b[0] < a[0]:
            c = [b[0] - dv[0] * length, b[1] - dv[1] * length]
        else:
            c = [b[0] + dv[0] * length, b[1] + dv[1] * length]
        return c

    
    #funtion to find midpoint coordinates between two points
    def midpoint(first, last):
        first = np.array(first)
        last = np.array(last)
        return [(first[0]+last[0])/2, (first[1]+last[1])/2]

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
    try:
        for human in humans: 
            # save coordinates of shoulder, hip, and elbow & neck
            neck_coord = [human.body_parts[1].x, human.body_parts[1].y]
            right_shoulder = [human.body_parts[2].x, human.body_parts[2].y]
            right_elbow = [human.body_parts[3].x, human.body_parts[3].y]
            right_hip = [human.body_parts[8].x, human.body_parts[8].y]
            left_shoulder = [human.body_parts[5].x, human.body_parts[5].y]
            left_elbow = [human.body_parts[6].x, human.body_parts[6].y]
            left_hip = [human.body_parts[11].x, human.body_parts[11].y]
    
        #find perpendicular coordinates and calculate the angle
        pc = perpCoord(neck_coord, right_shoulder)
        left_pc = perpCoord(neck_coord, left_shoulder)
        jointAngle = calculate_angle(pc, right_shoulder, right_elbow)
        formatted_angle = "{:.2f}".format(jointAngle)
        left_jointAngle = calculate_angle(left_pc, left_shoulder, left_elbow)
        left_formatted_angle = "{:.2f}".format(left_jointAngle)

        #test visualizing angle on image
        img_h, img_w = image.shape[:2]
        cv2.putText(image, formatted_angle,
                tuple(np.multiply(right_shoulder, [img_w, img_h]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 1,
                cv2.LINE_AA
            )
        cv2.putText(image, left_formatted_angle,
                tuple(np.multiply(left_shoulder, [img_w, img_h]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 1,
                cv2.LINE_AA
            )
        #visualize perpendicular coordinate
        #cv2.circle(image,tuple(np.multiply(pc, [img_w, img_h]).astype(int)),3,(0,0,255),thickness=3,lineType=8,shift=0)
        #cv2.circle(image,tuple(np.multiply(left_pc, [img_w, img_h]).astype(int)),3,(0,0,255),thickness=3,lineType=8,shift=0)
        #test midpoint
        mid = midpoint(right_hip, left_hip)
        cv2.circle(
                    image,
                    tuple(np.multiply(mid, [img_w, img_h]).astype(int)),
                    3,
                    (0,255,255),
                    thickness=3,
                    lineType=8,
                    shift=0,
                )
        #Draw spine
        cv2.line(
                    image,
                    tuple(np.multiply(neck_coord, [img_w, img_h]).astype(int)),
                    tuple(np.multiply(mid, [img_w, img_h]).astype(int)),
                    (0,255,255),
                    3
                )
        #find center of gravity
        cg = midpoint(neck_coord, mid)
        cv2.circle(
                    image,
                    tuple(np.multiply(cg, [img_w, img_h]).astype(int)),
                    3,
                    (255,0,255),
                    thickness=3,
                    lineType=8,
                    shift=0,
                )
    except:
        pass
    
    cv2.imshow('tf-pose-estimation result', image)
    cv2.waitKey()

    logger.info('3d test')
    poseLifting = Prob3dPose('./deep-lifting/data/saved_sessions/prob_model/prob_model_params.mat')

    