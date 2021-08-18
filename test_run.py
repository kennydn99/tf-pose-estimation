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
from deeplifting.packages.lifting.utils.draw import plot_pose
#from deeplifting.applications.demo import display_results

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
        a = np.array(a) #elbow
        b = np.array(b) #shoulder
        # find length between a and b
        length = np.linalg.norm(a-b)
        #delta x and delta y
        dv = [b[0] - a[0], b[1] - a[1]]
        return [b[0], b[1] + length]
    
    #funtion to find midpoint coordinates between two points
    def midpoint(first, last):
        first = np.array(first)
        last = np.array(last)
        return [(first[0]+last[0])/2, (first[1]+last[1])/2]

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
    try:
        for human in humans:
            #save coordinates of shoulder, hip, and elbow & neck
            neck = [human.body_parts[1].x, human.body_parts[1].y]
            RS = [human.body_parts[2].x, human.body_parts[2].y]
            RE = [human.body_parts[3].x, human.body_parts[3].y]
            RH = [human.body_parts[8].x, human.body_parts[8].y]
            LS = [human.body_parts[5].x, human.body_parts[5].y]
            LE = [human.body_parts[6].x, human.body_parts[6].y]
            LH = [human.body_parts[11].x, human.body_parts[11].y]
            
        img_h, img_w = image.shape[:2]
        #Calculate & visualize the angle for right and left shoulder
        right_pc = perpCoord(RE, RS)
        left_pc = perpCoord(LE, LS)
        right_angle = math.ceil(calculate_angle(right_pc, RS, RE))
        left_angle = math.ceil(calculate_angle(left_pc, LS, LE))
        cv2.putText(
                    image, str(right_angle),
                    tuple(np.multiply(RS, [img_w, img_h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1,
                    cv2.LINE_AA
                )
        cv2.putText(
                    image, str(left_angle),
                    tuple(np.multiply(LS, [img_w, img_h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1,
                    cv2.LINE_AA
                )
    except:
        pass
        
    try:
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
    
    cv2.imshow('tf-pose-estimation result', image)
    cv2.waitKey()

    logger.info('3d testing')
    poseLifting = Prob3dPose('./deeplifting/data/saved_sessions/prob_model/prob_model_params.mat')

    image_h, image_w = image.shape[:2]
    default_w = 640
    default_h = 480

    pose_2d_mpiis = []
    visibilities = []
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        pose_2d_mpiis.append([(int(x * default_w + 0.5), int(y * default_h + 0.5)) for x,y in pose_2d_mpii])
        visibilities.append(visibility)

    pose_2d_mpiis = np.array(pose_2d_mpiis)
    visibilities = np.array(visibilities)
    transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)
    
    #print 3d keypoints
    pose_3dqt = np.array(pose_3d[0].transpose())
    for p in pose_3dqt:
        print(p)

    #display_results(image, pose_2d_mpiis, visibilities, pose_3d)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title("Result")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        '''bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        bgimg = cv2.resize(
            bgimg,
            (e.heatMat.shape[1], e.heatMat.shape[0]),
            interpolation=cv2.INTER_AREA,
        )'''

        # show network output
        a = fig.add_subplot(2, 2, 2)
        #plt.imshow(bgimg, alpha=0.5)
        tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = e.pafMat.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title("Vectormap-x")
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title("Vectormap-y")
        # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
        #plt.show()

        #plot 3d points and show
        for i, single_3d in enumerate(pose_3d):
            plot_pose(single_3d)
        plt.show()

    except Exception as e:
        logger.warning("matplotlib error, %s" % e)
    