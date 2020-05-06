# PyTorch-YOLOv3
**해당 repo는 Erik Linder-Norén님의 [repo](https://github.com/eriklindernoren/PyTorch-YOLOv3)의 수정버전임을 알려드립니다.**

**xmltotxt의 경우 Isabek Tashiev님의 [repo](https://github.com/Isabek/XmlToTxt)를 그대로 사용하였습니다.**

**실제 implemetation시 오류와 불편한 점을 해결하고자 처리한 결과를 공유합니다.**

**자세한 설명은 각 repo에가시면 더 자세히 알 수 있습니다.**

### ※기본 욜로 버전을 v3에서 v4로 변경※

~~Erik Linder-Norén님은 GAN때도 많이 배웠는데 여기서도~~

## Purpose

해당 repo의 주 내용은 윈도우 환경에서 정상적으로 사용하기 위함입니다.

존재하는 버전관련 이슈도 함께 수정하였습니다.

## Main Contribution

### 1. bash파일을 py파일로 변환

-   config, data 폴더의 bash파일을 py파일로 변환하여 윈도우 환경에서도 활용가능하게 변경
-   데이터를 준비하고 위 명령어들을 한 번에 처리할 수 있도록 py파일 추가

### utils/logger 버전 업데이트

-   tf1버전으로 작성된 파일을 2버전에서도 사용할 수 있도록 변경

```python
# add tf.compat.v1.disable_eager_execution()

# tf.Summary => tf.compat.v1.Summary
```

### 3. utils/utils내에서 pytorch 버전 변경에 따른 Tensor 유형 변경

```python
# ByteTensor => BoolTensor
```

### 4. detect/train/test에서 use_custom 옵션 설정하여 한 번에 처리 하도록 추가

-   general purpose와는 거리가 있을 수 있음

### 5. video.py 추가

-   기존 image detect만 지원하던 것에서 video/cam을 활용할 수 있도록 파일 추가

### 6. video작업시 캡쳐 및 비디오 저장 기능 추가

-   결과는 output 폴더에 저장

-   **space**를 누르면 일시정지

-   **s**를 누르면 캡쳐

-   **r**을 누르면 녹화

-   **t**를 누르면 녹화 중지

### 7. Models.py

-   batchnorm을 하이퍼 파라미터에 따라서 적용하도록 변경

```bash
-   momentum=0.9 -> momentum=float(hyperparams["momentum"]
-   eps=1e-5 -> eps=float(hyperparams['decay']
```

-   Mish activation 추가

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

-   Darknet파일을 받기 위해 wget을 사용할 수 없으므로 직접 url을 입력하여 가져옵니다.
-   해당 파일을 weights 폴더에위치시킵니다.

```bash
# for v4
# yolov4.conv.137
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

-   pretrained모형을 받았으므로 바로 detect할 수 있습니다.
-   dadta/sample_images에 적용하고 싶은 image 파일들을 위치시키고 아래 명령어를 치시면 결과를 보실 수 있습니다.

```bash
$ python detect.py
```

-   결과 파일은 output폴더에 생성됩니다.

### Video/Cam

-   해당 모형을 동영상에 적용하는 추가사항입니다.
-   video파일의 경우 하나의 파일만 대상으로 확인하였습니다.
-   data/sample_videos에 비디오 파일을 위치시키고 아래 명령어를 입력합니다.

```bash
$ python video.py
```

-   cam을 사용하여 해당 기능을 사용하시려면 아래와 같이 명령어를 수정해줍니다.

```bash
$ python video.py --cam True
```

## Train on Custom Dataset

-   bash를 사용하는 부분에 대하여 수정된 부분입니다.

### Prepare Dataset

-   훈련에 사용 할 데이터셋을 준비합니다.
-   데이터셋은 image파일과 label파일이 필요합니다.

#### images

-   이미지 파일은 data/custom/images에 위치시킵니다.

#### labels

-   label파일의 경우 아래와 같은 형식의 txt파일입니다.

```
0 0.512144 0.413089 0.897571 0.733286
```

-   훈련을 위한 파일 중 xml파일을 가지고 계실경우 xmltotxt를 활용합니다.
-   classes.txt파일에 사용하는 데이터의 class들을 입력하고 마지막에는 빈 칸을 둡니다.
-   xmltotxt/xml에 해당 xml label을 위치시키고 아래 명령어를 입력해 줍니다.
-   다시 한 번 언급하자면 해당 파일들은 Isabek Tashiev님의 repo입니다.

```bash
$ python xmltotxt.py 
```

-   이 후 out폴더에 txt들을 label로 위치시켜주시면 됩니다.

-   마지막으로 data/custom/classes.names에도 같은 내용을 입력해줍니다.
    -   **마지막 칸을 비워둬야 정확히 인식합니다.**

### Build Config

-   개별적으로 bash명령을 통해 수행되는 작업을 py파일로 만들고 통합하였습니다.
-   위 준비사항이 끝나면 아래 명령어를 통해 나머지 config등에 대한 작업을 수행하실 수 있습니다.

```bash
$ python build_custom.py -n <class 개수>
```

-   위 명령어는 yolov3-custom.cfg파일을 만들고 
-   data폴더의 train.txt, valid.txt를 자동으로 생성합니다.

## Train
-   이제 아래 명령어로 trian시키 실 수 있습니다.

```bash
$ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
```

-   원 글처럼 `--pretrained_weights weights/darknet53.conv.74` 을 추가하여 pretrained_model를 기반으로 훈련시킬 수 있습니다.

-   위 명령어가 잘 생각나지 않으시면 아래와 같이 입력하시면 됩니다.
-   이 때 weight는 train이후 checkpoint의 마지막 결과를 받아오게 설정하였으므로 필요하시면 변경하시면됩니다.

```bash
$ python train.py --use_custom True
```

## Detect/Test

-   지금 까지 모형을 준비하여 train시키셨다면 결과를 활용할 차례입니다.
-   아래 옵션처럼 `use_custom True`를 사용하여 편하게 해당 모형을 활용할 수 있습니다.

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
