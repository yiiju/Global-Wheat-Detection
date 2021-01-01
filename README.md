# Global Wheat Detection

A kaggle competition, [Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection), to detect the wheat head.

## Introduce

Apply Faster R-CNN, YOLOv3 and YOLOv4-CSP in Scaled-YOLOv4 as the
model architecture.

Ensemble four models, Faster RNN, YOLOv3, YOLOv4-CSP and YOLOv4-CSP with pseudo-labeling by using weighted boxes fusion.

The pseudo-labeling is using the testing result to fine tune the model by 1 or 2 epoch.

The best performance in **private leaderboard is 0.6597** by using YOLOv4-CSP with pseudo-labeling.

## Hardware
Ubuntu 18.04 LTS

Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

1x GeForce RTX 2080 Ti

## Set Up
### Install Dependency
All requirements is detailed in requirements.txt.

    $ pip install -r requirements.txt

### YOLOv3

Using mmdetection.

Follow the [Installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) in mmdetection to set up the environment.
    
Move the wheat_mmdetection/configs/wheat/ folder into mmdetection/configs/.

### Coding Style
Use PEP8 guidelines.

    $ pycodestyle *.py

## Dataset
The data directory is structured as:
```
└── input 
    ├── test ─ test images
    ├── train ─ 3,434 training images
    ├── sample_submission.csv ─ the sample submission format
    └── train.csv - training annotations in csv format
```

## Train
Train in YOLOv3. (The root is in the mmdetection)

    $ python3 tools/train.py configs/wheat/pre_yolov3_1024_norm.py --gpu-ids 8 --work-dir "work_dirs/data_gamma_yolov3_1024"

Argument
 - `--gpu-ids` the ids of gpus to use
 - `--work-dir` the path to store the checkpoints and config setting

## Inference (on kaggle)
[YOLOv3](./inference_kaggle/YOLOv3.ipynb)

[YOLOv4-csp](./inference_kaggle/)

[YOLOv4-csp with pseudo-labeled from test data](./inference_kaggle/)

[Faster RCNN](./inference_kaggle/)

[Ensemble four models](./inference_kaggle/ensemble.ipynb)

## Citation
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}

@article{wbf2019,
  title={Weighted Boxes Fusion: ensembling boxes for object detection models},
  author={Solovyev, Roman and Wang, Weimin and Gabruseva, Tatiana},
  journal={arXiv preprint arXiv:1910.13302},
  year={2019}
}
```