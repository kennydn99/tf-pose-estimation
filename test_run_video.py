import argparse
import logging
import time
import math
import cv2
import numpy as np


from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

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
    ret_val, image = cam.read()
    logger.info("cam image=%dx%d" % (image.shape[1], image.shape[0]))

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

    while True:
        ret_val, image = cam.read()
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
                RS = [human.body_parts[2].x, human.body_parts[2].y]
                RE = [human.body_parts[3].x, human.body_parts[3].y]
                RH = [human.body_parts[8].x, human.body_parts[8].y]
                LS = [human.body_parts[5].x, human.body_parts[5].y]
                LE = [human.body_parts[6].x, human.body_parts[6].y]
                LH = [human.body_parts[11].x, human.body_parts[11].y]
            
            img_h, img_w = image.shape[:2]
            #Calculate & visualize the angle for right and left shoulder
            right_pc = perpCoord(neck, RS)
            left_pc = perpCoord(neck, LS)
            right_angle = "{:.2f}".format(calculate_angle(right_pc, RS, RE))
            left_angle = "{:.2f}".format(calculate_angle(left_pc, LS, LE))
            cv2.putText(
                        image, right_angle,
                        tuple(np.multiply(RS, [img_w, img_h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 1,
                        cv2.LINE_AA
                    )
            cv2.putText(
                        image, left_angle,
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
                    (0,255,255),
                    3
                )
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
        if cv2.waitKey(1) == 27:
            break
        logger.debug("finished+")

    cv2.destroyAllWindows()