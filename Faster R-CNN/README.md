# Faster R-CNN

Our Faster R-CNN method contain two part, one is original Faster R-CNN, another is training with heavy augmentation and test time augmentation (TTA)
Our implementation is base on kaggle notebook:

- [Pytorch Starter - FasterRCNN Train](https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train)
- [Competition metric details + script](https://www.kaggle.com/pestipeti/competition-metric-details-script)
- [Pytorch Starter - FasterRCNN Inference](https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-inference)
- [[WBF over TTA][Single Model] Fasterrcnn Resnest](https://www.kaggle.com/whurobin/wbf-over-tta-single-model-fasterrcnn-resnest)
- [Awesome Augmentation](https://www.kaggle.com/nvnnghia/awesome-augmentation)

Due to competition restrictions, we need to infer on kaggle. In our implementation, we train Faster R-CNN on local, and infer the testing data on kaggle notebook.


## Global Wheat Detection dataset
you can download dataset from [here](https://www.kaggle.com/c/global-wheat-detection/data).

And let file position like 
```
+- input
  +- global-wheat-detection
     | train.csv
     | sample_submission.csv
     +-train
     +-test
  
+- Faster R-CNN
   | Faster R-CNN training.py
   | Faster R-CNN aug.py
```

## Training

```
python Faster R-CNN training.py
```

or 

```
python Faster R-CNN aug.py
```

## Inference

Testing on [kaggle](https://www.kaggle.com/c/global-wheat-detection/overview) with `faster-r-cnn-inferenece.ipynb` or `faster-r-cnn-inferenece-aug.ipynb`

