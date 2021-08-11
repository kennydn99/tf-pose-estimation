import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger("TfPoseEstimator-Video")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tf-pose-estimation Video")
    parser.add_argument("--video", type=str, default="")
    parser.add_argument(
        "--resolution",
        type=str,
        default="432x368",
        help="network input resolution. default=432x368",
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
        "--showBG", type=bool, default=True, help="False to show skeleton only."
    )
    args = parser.parse_args()

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
        #calculate length bewteen a & b
        a = np.array(a)
        b = np.array(b)
        # find length between a and b
        length = np.linalg.norm(a-b)
        print('len:', length)
        #get direction vector = [delta x, delta y]
        dv = [b[0] - a[0], b[1] - a[1]]
        if dv[0] == 0 or dv[1] == 0:
            c = [b[0], b[1] + length]
            return c, dv
        #get magnitude
        var = math.sqrt(dv[0]*dv[0] + dv[1]*dv[1])
        dv[0] = dv[0]/var
        dv[1] = dv[1]/var
        print ('dv:', dv)
        #invert direction vector coordinate and swap
        dv[0], dv[1] = -dv[1], dv[0]
        print('swap dv:', dv)
        # new line starting at b pointing in direction of dv
        c = [b[0] - dv[0] * length, b[1] - dv[1] * length]
        return c, dv

    logger.debug("initialization %s : %s" % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file")
        
    while cap.isOpened():
        ret_val, image = cap.read()
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        humans = e.inference(image)
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        
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

    cv2.destroyAllWindows()
logger.debug("finished+")
