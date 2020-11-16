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
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="data/sample_videos/sample.mp4", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov4.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov4.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.1, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--cam", type=bool, default=False, help="use cam instead of video")
    parser.add_argument("--use_custom", type=bool, default=False, help="use custom trained weight")
    parser.add_argument("--output_dir", type=str, default='output/', help="dir to stroe recorded video or snapshot")
    parser.add_argument("--conf", action='store_true', help="add conf score in the label")
    opt = parser.parse_args()

    # Use custom weight
    if opt.use_custom:
        opt.model_def = 'config/yolov4-custom.cfg'
        opt.class_path = 'data/custom/classes.names'
        ls = sorted(os.listdir('./weights/custom'))
        if len(ls) > 0:
            opt.weights_path = 'weights/custom/'+ls[-1]
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
    print('###########################\n# press space to pause\n# press s to capture\n# press r to record\n# press t to finish recording\n###########################')
    
    # To transform from numpy to tensor
    trfs = transforms.ToTensor()

    # cam/video option
    if opt.cam:
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(opt.video_path)

    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    record = False
    n_frame = cap.get(cv.CAP_PROP_FPS)

    # bbox color map
    cmap = plt.get_cmap("tab20b")
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    
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
                if opt.conf:
                    label = f'{classes[int(cls_pred)]} {round(float(cls_conf), 2)}'
                else:
                    label = f'{classes[int(cls_pred)]}'
                lt = min(2, round(0.05 * (-x1.numpy() + x2.numpy())))
                cv.rectangle(frame, # target image
                            (x1, y1), # one point of rectengle
                            (x2, y2), # diagonal direction of the above
                            colors[int(cls_pred)], # color
                            1, # thickness
                            cv.LINE_AA
                            )
                cv.putText(frame, # target images
                            label , # str
                            (x1, y1+lt*4), # location; (x1, y1) would not work
                            cv.FONT_HERSHEY_SIMPLEX, # font
                            lt/4, # font size
                            [255, 255, 255], # color
                            1, # thickness
                            lineType=cv.LINE_AA
                            )
        # show results
        cv.imshow('cam', frame) # window-name, frame
        key = cv.waitKey(10)

        if key & 0xFF == 27: # esc
            break
        elif key == 32: # space
            print('press nay key to resume')
            cv.waitKey() # pause
        elif key == ord('s'):
            fname = datetime.datetime.now().strftime('%m%d%H%M%S') + '.jpg'
            fname = opt.output_dir + fname
            cv.imwrite(fname, frame)
        elif key == ord('r') and not record:
            print('recording...')
            vname = datetime.datetime.now().strftime('%m%d%H%M%S') + '.avi'
            vname = opt.output_dir + vname
            writer = cv.VideoWriter(vname, fourcc, int(n_frame), (opt.img_size, opt.img_size)) # can change video size option
            record = True
        elif key == ord('t') and record:
            print('finish recording...')
            writer.release()
            record = False
        
        if record:
            writer.write(frame)

    cap.release()
    cv.destroyAllWindows()