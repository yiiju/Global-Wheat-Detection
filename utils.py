import csv
import json
import glob
import operator
import pandas as pd
from PIL import Image
from Ensemble.ensemble_boxes_wbf import *


def convert2COCO(out_file, in_file):
    annotations = []
    images = []
    imgCount = 0
    objCount = 0    
    lastImg = None

    with open(in_file, newline='') as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            if row['image_id'] != lastImg:
                images.append(dict(id=imgCount,
                                file_name=row['image_id'] + '.jpg',
                                height=int(row['height']),
                                width=int(row['width'])))
                lastImg = row['image_id']
                imgCount = imgCount + 1

            w = int(row['bbox'][1:-1].split(', ')[2].split('.')[0])
            h = int(row['bbox'][1:-1].split(', ')[3].split('.')[0])
            xmin = int(row['bbox'][1:-1].split(', ')[0].split('.')[0])
            ymin = int(row['bbox'][1:-1].split(', ')[1].split('.')[0])
            objectAnno = dict(image_id=imgCount-1,
                              id=objCount,
                              category_id=0,
                              bbox=[xmin, ymin, w, h],
                              area=w * h,
                              segmentation=[],
                              iscrowd=0)
            objCount = objCount + 1

            annotations.append(objectAnno)
        
        cocoJson = dict(images=images,
                        annotations=annotations,
                        categories=[{'id': 0, 'name': 'wheat'}])

        with open(out_file, 'w') as json_file:
            json.dump(cocoJson, json_file)


def testJson(out_file, in_file, img_path):
    annotations = []
    images = []
    imgCount = 0
    objCount = 0
    lastImg = None
    
    with open(in_file, newline='') as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            if row['image_id'] != lastImg:
                image = Image.open(img_path + row['image_id'] + '.jpg').convert('RGB')
                imgw, imgh = image.size
                
                images.append(dict(id=imgCount,
                                file_name=row['image_id'] + '.jpg',
                                height=imgh,
                                width=imgw))
                
                lastImg = row['image_id']
                imgCount = imgCount + 1

            for i in range(0, int(len(row['PredictionString'].split(' '))), 5):
                xmin = int(row['PredictionString'].split(' ')[i+1])
                ymin = int(row['PredictionString'].split(' ')[i+2])
                w = int(row['PredictionString'].split(' ')[i+3])
                h = int(row['PredictionString'].split(' ')[i+4])
                bbox = (xmin, ymin, w, h)
                objectAnno = dict(image_id=imgCount-1,
                                id=objCount,
                                category_id=0,
                                bbox=bbox,
                                area=w * h,
                                segmentation=[],
                                iscrowd=0)
                objCount = objCount + 1
            
                annotations.append(objectAnno)

        cocoJson = dict(images=images,
                        annotations=annotations,
                        categories=[{'id': 0, 'name': 'wheat'}])
        
        with open(out_file, 'w') as json_file:
            json.dump(cocoJson, json_file)


def fakeJson(out_file, img_path):
    annotations = []
    images = []
    imgCount = 0
    objCount = 0     
    for imabsname in glob.glob(img_path + '/*.jpg'):
        imname = imabsname.split('/')[-1]
    
        image = Image.open(imabsname).convert('RGB')
        imgw, imgh = image.size

        images.append(dict(id=imgCount,
                           file_name=imname,
                           height=imgh,
                           width=imgw))

        x_min, y_min, w, h = (0, 0, 0, 0)
        objectAnno = dict(image_id=imgCount,
                          id=objCount,
                          category_id=0,
                          bbox=[x_min, y_min, w, h],
                          area=w * h,
                          segmentation=[],
                          iscrowd=0)
        
        annotations.append(objectAnno)
        imgCount = imgCount + 1
        objCount = objCount + 1

        cocoJson = dict(images=images,
                        annotations=annotations,
                        categories=[{'id': 0, 'name': 'wheat'}])
        
        with open(out_file, 'w') as json_file:
            json.dump(cocoJson, json_file)


def mmForm2SubmitForm(in_json, test_json, out_file):
    with open(in_json) as f:
        in_json = json.load(f)
    with open(test_json) as tf:
        test_json = json.load(tf)
    imgList = []
    for idx in test_json['images']:
        imgList.append(idx['file_name'])

    out_list = []
    lastImgId = 0
    pred_strings = []
    for idx in range(len(in_json)):
        imgId = in_json[idx]['image_id']
        left = int(in_json[idx]['bbox'][0]) if int(in_json[idx]['bbox'][0]) > 0 else 0
        top = int(in_json[idx]['bbox'][1]) if int(in_json[idx]['bbox'][1]) > 0 else 0
        width = int(in_json[idx]['bbox'][2]) if int(in_json[idx]['bbox'][2]) > 0 else 0
        height = int(in_json[idx]['bbox'][3]) if int(in_json[idx]['bbox'][3]) > 0 else 0
        box = [left, top, width, height]
    
        if imgId == lastImgId:
            imageName = imgList[imgId].split('/')[-1].split('.')[0]
            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(in_json[idx]['score'], box[0], box[1], box[2], box[3]))
        else:
            # Save previous result
            result = {
                'image_id': imageName,
                'PredictionString': " ".join(pred_strings)
            }
            out_list.append(result)

            if imgId-lastImgId > 1:
                emptyNum = imgId - lastImgId - 1
                print(imgId, lastImgId)
                skipId = lastImgId + 1
                while emptyNum:
                    emptyNum = emptyNum - 1
                    imageName = imgList[skipId].split('/')[-1].split('.')[0]
                    result = {
                        'image_id': imageName,
                        'PredictionString': ""
                    }
                    out_list.append(result)
                    skipId = skipId + 1

            imageName = imgList[imgId].split('/')[-1].split('.')[0]
            pred_strings = []
            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(in_json[idx]['score'], box[0], box[1], box[2], box[3]))
        
        # Last one
        if idx == len(in_json)-1:
            result = {
                'image_id': imageName,
                'PredictionString': " ".join(pred_strings)
            }
            out_list.append(result)
        
        lastImgId = imgId

    test_df = pd.DataFrame(out_list, columns=['image_id', 'PredictionString'])
    test_df.to_csv(out_file, index=False)


def spike2wheatForm(img_path, tsv_path, out_file):
    out_list = []
    img_list = glob.glob(img_path + '/*.jpg')
    for img in img_list:
        image = Image.open(img).convert('RGB')
        imgw, imgh = image.size
        name = img.split('.jpg')[0].split('/')[-1]
        tsv_file = open(tsv_path + name + '.bboxes.tsv')
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for row in read_tsv:
            bbox_strings = "[{0}, {1}, {2}, {3}]".format(row[0], row[1], row[2], row[3])
            result = {
                'image_id': name,
                'width': imgw,
                'height': imgh,
                'bbox': bbox_strings,
                'source':'spike'
            }
            out_list.append(result)
    test_df = pd.DataFrame(out_list, columns=['image_id', 'width', 'height', 'bbox', 'source'])
    test_df.to_csv(out_file, index=False)

def combineCSV(csvpath, outcsv):
    csvfile = glob.glob(csvpath + '/*.csv')
    combined_csv = pd.concat([pd.read_csv(f) for f in csvfile])
    combined_csv.to_csv(outcsv, index=False)


def submission2traincsv(subcsv, img_folder, outcsv):
    out_list = []
    with open(subcsv, newline='') as csv_file:
        rows = csv.DictReader(csv_file)
        for row in rows:
            imgname = row['image_id']
            img = Image.open(img_folder + imgname + '.jpg').convert('RGB')
            imgw, imgh = img.size
            for i in range(0, int(len(row['PredictionString'].split(' '))), 5):
                xmin = int(row['PredictionString'].split(' ')[i+1])
                ymin = int(row['PredictionString'].split(' ')[i+2])
                w = int(row['PredictionString'].split(' ')[i+3])
                h = int(row['PredictionString'].split(' ')[i+4])
                bbox_strings = [xmin, ymin, w, h]
                result = {
                    'image_id': imgname,
                    'width': imgw,
                    'height': imgh,
                    'bbox': bbox_strings,
                    'source':'train'
                }
                out_list.append(result)
    test_df = pd.DataFrame(out_list, columns=['image_id', 'width', 'height', 'bbox', 'source'])
    test_df.to_csv(outcsv, index=False)

def conbine(subcsv1, subcsv2, outcsv):
    out_list = []
    csv1 = open(subcsv1)
    csv2 = open(subcsv2)
    next(csv1)
    next(csv2)
    rows1 = csv.reader(csv1)
    rows2 = csv.reader(csv2)
    sortedrow1 = sorted(rows1, key=operator.itemgetter(0))
    sortedrow2 = sorted(rows2, key=operator.itemgetter(0))
    for row1, row2 in zip(sortedrow1, sortedrow2):
        imgname = row1[0]
        bbox_strings1 = []
        bbox_strings2 = []
        score1 = []
        score2 = []
        label1 = []
        label2 = []
        pred_strings = []
        for i in range(0, int(len(row1[1].split(' '))), 5):
            score1.append(float(row1[1].split(' ')[i]))
            w = int(row1[1].split(' ')[i+3])
            h = int(row1[1].split(' ')[i+4])
            xmin = int(row1[1].split(' ')[i+1])
            ymin = int(row1[1].split(' ')[i+2])
            xmax = (xmin + w)
            ymax = (ymin + h)
            bbox_strings1.append([xmin / 1024, ymin / 1024, xmax / 1024, ymax / 1024])
            label1.append(1)
            
            score2.append(float(row2[1].split(' ')[i]))
            w = int(row2[1].split(' ')[i+3])
            h = int(row2[1].split(' ')[i+4])
            xmin = int(row2[1].split(' ')[i+1])
            ymin = int(row2[1].split(' ')[i+2])
            xmax = (xmin + w)
            ymax = (ymin + h)
            bbox_strings2.append([xmin / 1024, ymin / 1024, xmax / 1024, ymax / 1024])
            label2.append(1)

        score = [score1, score2]
        boxes_list = bbox_strings1, bbox_strings2
        label = [label1, label2]
        weights = [2, 1]
        
        iou_thr = 0.5
        skip_box_thr = 0.0001
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, score, label, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        for i in range(len(scores)):
            xmin = int(boxes[i][0] * 1024)
            ymin = int(boxes[i][1] * 1024)
            w = int((boxes[i][2] - boxes[i][0]) * 1024)
            h = int((boxes[i][3] - boxes[i][1]) * 1024)
            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(scores[i], xmin, ymin, w, h))

        result = {
                'image_id': imgname,
                'PredictionString': " ".join(pred_strings)
        }
        out_list.append(result)

    test_df = pd.DataFrame(out_list, columns=['image_id', 'PredictionString'])
    test_df.to_csv(outcsv, index=False)