from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import cv2 as cv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="data/sample_videos/sample.mp4", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--cam", type=bool, default=False, help="use cam instead of video")
    parser.add_argument("--use_custom", type=bool, default=False, help="trained weight")
    opt = parser.parse_args()
    print(opt)

    # Use custom weight
    if opt.use_custom:
        opt.model_def = 'config/yolov3-custom.cfg'
        ls = sorted(os.listdir('./checkpoints'))
        if len(ls) > 0:
            opt.weights_path = 'checkpoints/'+ls[-1]
        opt.class_path = 'data/custom/classes.names'
        print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    # Eval mode
    model.eval()

    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    # To transform from numpy to tensor
    trfs = transforms.ToTensor()

    # cam/video option
    if opt.cam:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(opt.video_path)

    # Real-time detection
    while cap.isOpened():
        ret, frame = cap.read()
        # Preprocessing especially size
        frame = cv.resize(frame, (opt.img_size, opt.img_size))
        input_frame = trfs(frame)
        input_frame = input_frame.unsqueeze(0)
        input_frame = Variable(input_frame.type(Tensor)) 

        # Predict detections
        with torch.no_grad():
            detections = model(input_frame)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)[0] # Detections: x1, y1, x2, y2, conf, cls_conf, cls_pred
            
        # Draw boxed
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                cv.rectangle(frame, # target image
                            (x1, y1), # one point of rectengle
                            (x2, y2), # diagonal direction of the above
                            (0, 0, 255), # color
                            3 # thickness
                            )
                cv.putText(frame, # target images
                            classes[int(cls_pred)], # str
                            (x1+10, y1+10), # location; (x1, y1) would not work
                            cv.FONT_HERSHEY_SIMPLEX, # font
                            1, # font size
                            (255, 0, 0), # color
                            2 # thickness
                            )

        # show results
        cv.imshow('cam', frame)
        cv.waitKey(10)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()