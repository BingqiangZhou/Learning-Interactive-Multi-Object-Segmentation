import os
import glob
from re import X
import cv2 as cv
import numpy as np
from PIL import Image
import pandas as pd
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

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        return 1
    return 0

config_file_path = './config/t.yaml'
config = Config(config_file_path).get_configs()
model_file_path = './models/best_mean_iou_epoch.pkl'

net = OurNet(config, model_file_path)

datasets = ["GrabCut", "Berkeley"]

# ----- dataset dir -----
dataset_root_dir = '/home/guiyan/workspaces/datasets'


for dataset_name in datasets:
    dataset_dir = os.path.join(dataset_root_dir, dataset_name)
    image_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, "masks")
    interactives_dir = f"/home/guiyan/workspaces/bingqiangzhou/iis/data/interactives_{dataset_name}"

    out_dir = f'./seg_result_{dataset_name}_0712_best'
    mkdir(out_dir)

    excel_path = os.path.join(out_dir, 'out.xlsx')
    writer = pd.ExcelWriter(excel_path)

    image_paths = glob.glob(image_dir + "/*")

    max_num_points = 20
    for j in range(max_num_points):
        result_dir = os.path.join(out_dir, f"points_{j+1}")
        mkdir(result_dir)

        data_list = []
        mean_iou_list = []
        for path in tqdm(image_paths, desc=f"{dataset_name}-points-{j+1}"):
            image_np = np.array(Image.open(path))
            name = os.path.splitext(os.path.split(path)[1])[0]

            gt = np.zeros(image_np.shape[:2])
            if dataset_name == "Berkeley":
                label_np = np.array(Image.open(os.path.join(masks_dir, name+'.png')))
                gt[np.all(label_np == [255, 255, 255], axis=-1)] = 1
            else: # GrabCut
                label_np = np.array(Image.open(os.path.join(masks_dir, name+'.bmp')))
                gt[label_np > 0] = 1 # 数据集中只有单个对象
            

            interactive_map = np.array(Image.open(f"{interactives_dir}/{name}_fg_interactivemap_point{j+1}.png"))
            
            interactive_map[interactive_map == 255] = 1
            distance_map = bwdist(interactive_map)
            distance_map[distance_map > 255] = 255

            outputs, spend_time = net.predict_from_numpy(image_np, distance_map[:,:, np.newaxis])
            outputs = np.argmax(outputs, axis=-1) # (h, w, 2) -> (h, w)
            
            iou_v = iou(outputs, gt)
            bg_iou_v = iou((outputs == 0).astype(np.uint8), (gt == 0).astype(np.uint8))
            f1_score_v = f1_score(outputs, gt)
            bg_f1_score_v = iou((outputs == 0).astype(np.uint8), (gt == 0).astype(np.uint8))

            cv.imwrite(os.path.join(result_dir, f"{name}_point{j+1}_iou{iou_v}_f1score{f1_score_v}_time{spend_time}.png"), outputs*255)
            
            mean_iou = (iou_v + bg_iou_v) / 2
            data_list.append([name, iou_v, bg_iou_v, mean_iou, f1_score_v, bg_f1_score_v, (f1_score_v + bg_f1_score_v) / 2, spend_time])
            
            mean_iou_list.append(mean_iou)
        data_list = pd.DataFrame(data_list, #index=file_names, 
                                    columns=["image name" , "iou", "bg iou", "mean iou", "f1-score", "bg f1-score", "mean f1-score", "speed time"])
        data_list.to_excel(writer, sheet_name=f"point_{j+1}",  float_format="%.6f", index=False)
        writer.save()
        # print(np.mean(mean_iou_list))
    writer.close()
        
