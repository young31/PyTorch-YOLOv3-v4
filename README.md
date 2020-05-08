# PyTorch-YOLOv3

**This repo is revision version of Erik Linder-Norén's [repo](https://github.com/eriklindernoren/PyTorch-YOLOv3)**

**xmltotxt is originated from Isabek Tashiev's [repo](https://github.com/Isabek/XmlToTxt)**

**As I use origin version, I got several errors and I want to share what I do to overcome**

**If you are interested, visit origin repository above**

### ※Default Version changes yolov3 to yolov4※

한국어 버전은 [여기](https://github.com/young31/PyTorch-YOLOv3/blob/master/README-kr.md)

## Purpose

I made this to use in Windows environment

Some version difference issues are revised

## Main Contribution

### 1. Convert from .sh file to .py file

-   .sh files in config, data folder are converted for window users to use
-   additional file to apply above easily can be used 

### 2. In utils/logger Tensorflow version issues

-   tf1 version codes are converted to tf2 usage

```python
# add tf.compat.v1.disable_eager_execution()

# tf.Summary => tf.compat.v1.Summary
```

### 3. In utils/utils PyTorch version issue

```python
# ByteTensor => BoolTensor
```

### 4. Add use_custom option in detect/train/test for convenience

-   Not general purpose

### 5. Add video.py 

-   Now you can apply models with video/cam 

### 6. Add capture and recording

-   output will be saved in output folder
-   press **space** to pause
-   press **s** to capture
-   press **r** to record
-   press **t** to finish recording

### 7. Models.py(2020.05.06)

-   insert hyperparameter value

```bash
-   momentum=0.9 -> momentum=float(hyperparams["momentum"]
-   eps=1e-5 -> eps=float(hyperparams['decay']
```

-   add mish activation

```python
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
        
elif module_def["activation"] == "mish":
	modules.add_module(f"mish_{module_i}", Mish())
```


## Download Prerequisite

-   As not using wget, manually download files via enter urls
-   Files below should be located in weights folder

```bash
# for v4
# yolov4.conv.137
https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view
# weights
https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view
```
#############################################################################

```bash
# for v3
# darknet
https://pjreddie.com/media/files/darknet53.conv.74
# yolo
https://pjreddie.com/media/files/yolov3-tiny.weights
# yolo-tiny
https://pjreddie.com/media/files/yolov3-tiny.weights
```

## Quick Start

### Images

-   After done above, you can use pretrained model immedeately
-   After locate images in data/sample_images, and apply below

```bash
$ python detect.py
```

-   Annotated output will be cound in output folder

### Video/Cam

-   This is added function
-   Only per a video now and storing is not realized yet
-   Enter the command below and you can see pop-up

```bash
$ python video.py
```

-   If someone wants to use own webcam, command below would be useful

```bash
$ python video.py --cam True
```

## Train on Custom Dataset

### Prepare Dataset

-   Dataset should have images and labels(annotations)

#### images

-   Put images in data/custom/images folder

#### labels

-   Put labels in data/labels folder with *.txt 
-   If you have xml version, use xmltotxt

-   Write down your datasets' labels in classes.txt and enter the command below

```bash
$ python xmltotxt.py 
```

-   Outputs can be found in output folder and copy and paste them
-   Write down your classes in custom/classes.names
    -   **Should have one empty space**

### Build Config

-   I merge some functions
-   After preparing datsets properly, enter the code below
    -   It will make yolov3-custom.py and trian.txt, valid.txt for training

```bash
$ python build_custom.py -n <number of classes>
```

## Train

-   Only left thing is training model

```bash
$ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
```

-   You can add  `--pretrained_weights weights/darknet53.conv.74` to start with pretrained model
-   Command below acts same except `model_def`
    -   It use the latest training model in checkpoint folder

```bash
$ python train.py --use_custom True
```

## Detect/Test

-   After training, detect images and videos
    -   `__use_custom` option will use the latest trained model

```bash
# for images
$ python detect.py --use_custom True
# for video/cam
$ python video.py --use_custom True --cam True
```

## Credit

### YOLOv3: An Incremental Improvement

_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```