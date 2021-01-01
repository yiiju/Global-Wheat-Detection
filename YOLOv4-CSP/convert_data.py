import os
import shutil
import math
import pandas as pd
import numpy as np
import re
import cv2
    

all_classes = {'wheat': 0}
train_img = "yolo_data"
train_annotation = "../input/global-wheat-detection/train"
yolo_path = "yolo_data/"
write_train_txt = './data/train.txt'
write_val_txt = './data/val.txt'


if not os.path.exists(yolo_path):
    os.mkdir(yolo_path)

if os.path.exists(write_train_txt):
    file=open(write_train_txt, 'w')

if os.path.exists(write_val_txt):
    file=open(write_val_txt, 'w')

train_df = pd.read_csv('../input/global-wheat-detection/' + 'train.csv')
train_df['x'], train_df['y'], train_df['w'], train_df['h'] = -1, -1, -1, -1


def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

image_ids = train_df['image_id'].unique()

img_w = 1024
img_h = 1024
#'''
for img_name in image_ids:

    #img = cv2.imread('./yolo_data/' + img_name + '.jpg')
    #img_h, img_w = img.shape[:2]
    img_path = os.path.join(train_img, img_name)
    df = train_df[train_df['image_id'].isin([img_name])]   
    i = 0
    img_info = []
    for index, row in df.iterrows():
        xmin = row['x']
        ymin = row['y']
        raww = row['w']
        rawh = row['h']

        x = (xmin + (raww/2)) * 1.0 / img_w
        y = (ymin + (rawh/2)) * 1.0 / img_h
        w = raww * 1.0 / img_w
        h = rawh * 1.0 / img_h 
        objclass = 0
        img_info.append(' '.join([str(objclass), str(x),str(y),str(w),str(h)]))

    with open(yolo_path + str(img_name) + '.txt', 'w') as f:
        f.write('\n'.join(img_info))

print('the file is processed')

# create train and val txt
path = train_img#os.path.join(train_img, yolo_path)
datasets = []
for idx in image_ids:
    if not idx.endswith('.txt'):
        path = train_img + '/' + idx + '.jpg'
        datasets.append(path)

with open(write_train_txt, 'a') as f:
    f.write('\n'.join(datasets))

with open(write_val_txt, 'a') as f:
    f.write('\n'.join(datasets))


test_df = pd.read_csv('../input/global-wheat-detection/' + 'sample_submission.csv')
image_ids = test_df['image_id'].unique()

path = 'test/'
datasets = []
for idx in image_ids:
    if not idx.endswith('.txt'):
        img_path = path  + idx + '.jpg'
        datasets.append(img_path)

with open('./data/test.txt', 'w') as f:
    f.write('\n'.join(datasets))