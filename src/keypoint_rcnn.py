#!/usr/bin/env python3

import torch
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image

from cv_bridge import CvBridge, CvBridgeError
from struct import *
import rospy
from std_msgs.msg import *
from sensor_msgs.msg import *

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class keypoint_rcnn():
    def __init__(self):
        self.cv_bridge = CvBridge() 
        # Publisher
        self.image_pub = rospy.Publisher("/keypoint_rcnn/predict", Image, queue_size=1)

        # Subscriber
        rospy.Subscriber('/camera_mid/color/image_raw', Image, self.image_callback, queue_size=1)

        # Inference with a keypoint detection model
        self.cfg = get_cfg()   # get a fresh new config
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

    def image_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        outputs = self.predictor(cv_image)
        v = Visualizer(cv_image[:,:,::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(outputs)
        self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(out.get_image()[:, :, ::-1], "bgr8"))
        print("Detected 1 frame !!!")

    def onShutdown(self):
        rospy.loginfo("Shutdown.")

if __name__ == '__main__':
    # ros init
    rospy.init_node('keypoint_rcnn_node', anonymous=False)
    keypoint_rcnn = keypoint_rcnn()
    rospy.on_shutdown(keypoint_rcnn.onShutdown)
    rospy.spin()