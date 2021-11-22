import os
import time
import cv2 as cv
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

from config import Config
from net import OurNet

def bwdist(binary_mask):
    distance_map = ndimage.morphology.distance_transform_edt(1 - binary_mask)
    return distance_map 

def iou(binary_predict, binary_target, epsilon=1e-6):
    add_result = binary_predict + binary_target
    union = (add_result == 2).astype(np.uint8)
    intersection = add_result - union
    
    return np.sum(union) / (np.sum(intersection) + epsilon)

def f1_score(binary_predict, binary_target,  epsilon=1e-6):
    add_result = binary_predict + binary_target
    union = (add_result == 2).astype(np.uint8)
    precision = np.sum(union) /  (np.sum(binary_predict) + epsilon)
    recall = np.sum(union) /  (np.sum(binary_target) + epsilon)

    return 2 * (precision * recall) / (precision + recall + epsilon)

config_file_path = './config/t.yaml'
config = Config(config_file_path).get_configs()
model_file_path = './models/best_mean_iou_epoch.pkl'

net = OurNet(config, model_file_path)

# ----- dataset dir -----
voc_root_dir = '/home/guiyan/workspaces/datasets/voc2012/VOCdevkit/VOC2012'
mask_dir = os.path.join(voc_root_dir, 'SegmentationObject')
image_dir = os.path.join(voc_root_dir, 'JPEGImages')
splits_file = os.path.join(voc_root_dir, 'ImageSets/Segmentation/val.txt')
interactives_dir = "./interactives"

# ----- get all file name from val dataset -----
with open(os.path.join(splits_file), "r") as f:
    file_names = [x.strip() for x in f.readlines()]

# ----- output dir -----
out_dir = './seg_result_voc_mutli_object'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

excel_path = os.path.join(out_dir, 'out.xlsx')
writer = pd.ExcelWriter(excel_path)

max_num_points = 20
for j in range(max_num_points):
    result_dir = os.path.join(out_dir, f"points_{j+1}")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    data_list = []
    mean_iou_with_bg_in_dataset = []
    for name in tqdm(file_names, desc=f"points-{j+1}"):
        image_np = np.array(Image.open(os.path.join(image_dir, name+'.jpg')))
        label_np = np.array(Image.open(os.path.join(mask_dir, name+'.png')))
        label_np[label_np == 255] = 0
        
        ids = np.unique(label_np)
        nums_object = len(ids) - 1

        distance_maps = []
        for i in ids:
            if i == 0:
                continue

            interactive_map = np.array(Image.open(f"{interactives_dir}/{name}_fg_interactivemap_object{i}_point{j+1}.png"))
            
            interactive_map[interactive_map == 255] = 1
            distance_map = bwdist(interactive_map)
            distance_map[distance_map > 255] = 255
            distance_maps.append(distance_map)

        distance_maps = np.stack(distance_maps, axis=-1)
        outputs, spend_time = net.predict_from_numpy(image_np, distance_maps)
        outputs = np.argmax(outputs, axis=-1) # (h, w, n) -> (h, w)
        
        pred_bg = np.ones_like(outputs)
        ious = []
        f1_scores = []
        for i, v in enumerate(ids):
            if v == 0:
                continue
            gt = np.uint8(label_np == v)
            pred = np.uint8(outputs == i)
            pred_bg[pred > 0] = 0
            iou_v = iou(pred, gt)
            ious.append(iou_v)
            f1_score_v = f1_score(pred, gt)
            f1_scores.append(f1_score_v)
            cv.imwrite(os.path.join(result_dir, f"{name}_object{i}_point{j+1}_iou{iou_v}_f1score{f1_score_v}_time{spend_time}.png"), pred*255)
        
        mean_iou = np.mean(ious)
        mean_f1_score = np.mean(f1_scores)

        bg_gt = np.uint8(label_np == 0)
        iou_bg = iou(pred_bg, bg_gt)
        ious.append(iou_bg)
        f1_score_bg = f1_score(pred_bg, bg_gt)
        f1_scores.append(f1_score_bg)
        mean_iou_with_bg = np.mean(ious)
        mean_iou_with_bg_in_dataset.append(mean_iou_with_bg)
        mean_f1_score_with_bg = np.mean(f1_scores)

        for k in range(len(ious)):
            data_list.append([name, nums_object, k, ious[k], mean_iou, f1_scores[k], mean_f1_score, 
                                spend_time, spend_time / nums_object, 
                                iou_bg, mean_iou_with_bg, f1_score_bg, mean_f1_score_with_bg])
        plt.imsave(os.path.join(result_dir, f"{name}_point{j+1}_mean_iou{mean_iou}_mean_f1score{mean_f1_score}.png"), outputs)

    data_list = pd.DataFrame(data_list, #index=file_names, 
                                columns=["image name" , "num objects", "object index",  
                                "iou", "mean iou", "f1-score", "mean f1-score", "speed times", "mean times for object", 
                                "bg iou", "mean iou with bg", "bg f1-score", "mean f1-score with bg"])
    data_list.to_excel(writer, sheet_name=f"point_{j+1}",  float_format="%.6f", index=False)
    writer.save()
    # print(np.mean(mean_iou_with_bg_in_dataset))
writer.close()
