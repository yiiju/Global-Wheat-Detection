# YOLOv4-CSP

One of the versions of [Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)

Our training pipline base on [YOLOv4-CSP 訓練教學](https://medium.com/ching-i/scaledyolov4-yolov4-csp-%E8%A8%93%E7%B7%B4%E6%95%99%E5%AD%B8-ee091598e503).

## Global Wheat Detection dataset
You can download dataset from [here](https://www.kaggle.com/c/global-wheat-detection/data).


## Download YOLOv4-CSP

```
git clone https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp
```

And let file position like 
```
+- input
  +- global-wheat-detection
     | train.csv
     | sample_submission.csv
     +-train
     +-test
  
+- YOLOv4-CSP
   | ...
```

## Convert data to YOLO format
For training YOLOv4-CSP, we need to convert the data to the corresponding format.
The label of bounding box should normalize left top point, width and height to 0~1.
One image corresponds to one .txt file with label.
```
python convert.py
```
## Directory

After convert:
```
+- input
  +- global-wheat-detection
     | train.csv
     | sample_submission.csv
     +-train
     +-test
  
+- YOLOv4-CSP
   | train.py
   | test.py
   | detect.py
   | 
   +- data
   +- models
   +- utils 
   +- yolo_data
      | your image.jpg
      | your label.txt
        
```

## Download pre-trained weight
You can download pre-trained from [here](https://drive.google.com/file/d/1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL/view)

## Installation

Follow [YOLOv4-CSP](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp)
```
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov4_csp -it -v your_coco_path/:/coco/ -v your_code_path/:/yolo --shm-size=64g nvcr.io/nvidia/pytorch:20.06-py3

# install mish-cuda, if you use different pytorch version, you could try https://github.com/JunnYu/mish-cuda
cd /
git clone https://github.com/thomasbrandon/mish-cuda
cd mish-cuda
python setup.py build install

# go to code folder
cd /yolo
```

## Modify cfg

Modify width, height, filter number and class number
```
$ cp models/yolov4-csp.cfg models/yolov4-csp_1024.cfg
# check original parameter
$ sed -n -e 8p -e 9p -e 1022p -e 1029p -e 1131p -e 1138p -e 1240p -e 1247p models/yolov4-csp_416.cfg
```

Change width, height to 1024, and our dataset only has one class, so the filter number is (classes + 5)*3 = (1+5)*3 = 18

```
$ sed -i ‘8s/512/1024/’ models/yolov4-csp_1024.cfg
$ sed -i ‘9s/512/1024/’ models/yolov4-csp_1024.cfg
$ sed -i ‘1022s/255/18/’ models/yolov4-csp_1024.cfg
$ sed -i ‘1029s/80/1/’ models/yolov4-csp_1024.cfg
$ sed -i ‘1131s/255/18/’ models/yolov4-csp_1024.cfg
$ sed -i ‘1138s/80/1/’ models/yolov4-csp_1024.cfg
$ sed -i ‘1240s/255/18/’ models/yolov4-csp_1024.cfg
$ sed -i ‘1247s/80/1/’ models/yolov4-csp_1024.cfg
# check parameter afer modify
$ sed -n -e 8p -e 9p -e 1022p -e 1029p -e 1131p -e 1138p -e 1240p -e 1247p models/yolov4-csp_1024.cfg

```

## Create .yaml
like `./data/wheat.yaml`

```
train: ./data/train.txt  
val: ./data/val.txt  
test: ./data/test.txt  

# number of classes
nc: 1

# class names
names: ['wheat']

```

## Creat .name
like `./data/wheat.name`
```
wheat
```
## modify hyp.scratch.yaml

You can add more augmentation in `data/hyp.scratch.yaml`

## Training

```
sh sigle-train.sh
```
or 
```
sh multi-train.sh
```
or you can change to your own suitable parameters

## Testing 

Infer on kaggle by `yolov4-csp-inference.ipynb`
# Reference

[YOLOv4-CSP](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp)
[YOLOv4-CSP training tutorial](https://medium.com/ching-i/scaledyolov4-yolov4-csp-%E8%A8%93%E7%B7%B4%E6%95%99%E5%AD%B8-ee091598e503)
[YOLO data format](https://medium.com/ching-i/%E5%A6%82%E4%BD%95%E8%BD%89%E6%8F%9B%E7%82%BAyolo-txt%E6%A0%BC%E5%BC%8F-f1d193736e5c)

# Citation
```
@article{wang2020scaled,
  title={{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2011.08036},
  year={2020}
}
```