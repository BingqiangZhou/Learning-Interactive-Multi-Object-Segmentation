import os
import glob
from PIL import Image
import numpy as np

from codes.utils import set_rgb_color_map_by_name, color_map
from codes.stack import ArrayStack
from net import MOSNet

class Args():

    N = 9
    image_dir = './test_images'
    # image_dir = r'E:\Datasets\iis_datasets\VOCdevkit\VOC2012\JPEGImages'
#     image_dir = r'/home/guiyan/workspaces/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages'
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    color_map = set_rgb_color_map_by_name(N, zero_to_255=True)
    # color_map = color_map(32, rgb_or_bgr='rgb')
    object_nums_str = [str(i+1) for i in range(N)]
    
    cur_fig = None

    cur_image_index = 0
    max_image_index = len(image_paths)

    cur_image_name = os.path.splitext(os.path.split(image_paths[cur_image_index])[1])[0]
    source_image = np.array(Image.open(image_paths[cur_image_index]))
    show_image = source_image.copy()
    result = None
    embedding = None
    visual_attention_maps = None

    cur_object_index = 1
    points_list = ArrayStack(3)

    device_num = -1
    config_file_path = '../config/MOS_For_Demo.yaml'
    model_file_path = '../models/best_mean_iou_epoch.pkl'
    net = MOSNet(device_num=device_num, 
                config_file_path=config_file_path,
                model_file_path=model_file_path)
    auto_predict = True
