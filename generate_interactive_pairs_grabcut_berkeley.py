
import os
import glob
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt

def bwdist(binary_mask):
    distance_map = ndimage.morphology.distance_transform_edt(1 - binary_mask)
    return distance_map 

# ----- dataset dir -----
datasets_root_dir = r'iis_datasets'
datasets = ["GrabCut", "Berkeley"]

# ----- random sample parameters -----
max_num_points = 20
dmargin_interval=[0.05, 0.5]
d = 40
margin = 5
step = 10

for dataset_name in datasets:
    save_dir = f"./interactives_{dataset_name}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    dataset_mask_path = os.path.join(datasets_root_dir, dataset_name, "masks")

    masks_path = glob.glob(dataset_mask_path + "/*")


    for index, file_path in enumerate(masks_path):
        name = os.path.splitext(os.path.split(file_path)[1])[0]
        label = np.array(Image.open(file_path))
        binary_mask = np.zeros(label.shape[:2])
        if label.ndim > 2:
            binary_mask[np.all(label == [255, 255, 255], axis=-1)] = 1
        else:
            binary_mask[label > 0] = 1 # 数据集中只有单个对象
        
        fg_points_binary_map = np.zeros_like(binary_mask) # positive channel
        bg_points_binary_map = np.zeros_like(binary_mask) # negative channel
        
        max_margin = np.max(bwdist(1-binary_mask))
        
        bg_sample_region = np.int8((bwdist(binary_mask) < d) - binary_mask)
        original_bg_sample_region = np.int8((bwdist(1-bg_sample_region) > margin))

        for j in tqdm(range(max_num_points), desc=f'{dataset_name}-{name}'):
            
            # ----- sample points for foreground -----
            d_margin = np.random.randint(max_margin * dmargin_interval[0], max_margin * dmargin_interval[1] + 1)
            fg_sample_region = bwdist(1-binary_mask) > d_margin
            if j == 0:
                sample_map = np.random.rand(*(fg_sample_region.shape)) * fg_sample_region
                max_index = np.argmax(sample_map)
            else:
                max_index = np.argmax(bwdist(fg_points_binary_map) * fg_sample_region)
            # points_binary_map[index // binary_mask.shape[1], index % binary_mask.shape[1]] = 1
            x, y = np.unravel_index(max_index, sample_map.shape)
            fg_points_binary_map[x, y] = 1
            cv.imwrite(f"{save_dir}/{name}_fg_interactivemap_point{j+1}.png", fg_points_binary_map*255)

            # ----- sample points for background -----
            if j == 0:
                bg_sample_region = original_bg_sample_region
            else: #  j > 0
                bg_sample_region = np.int8(original_bg_sample_region + (bwdist(bg_points_binary_map) > step) == 2)
            sample_map = np.random.rand(*(bg_sample_region.shape)) * bg_sample_region
            max_value = np.max(sample_map)
            if max_value != 0:
                max_index = np.argmax(sample_map)
                # x = index / ncols , y = index % ncols
                # points_binary_map[index // binary_mask.shape[1], index % binary_mask.shape[1]] = 1
                x, y = np.unravel_index(max_index, sample_map.shape)
                bg_points_binary_map[x, y] = 1
            cv.imwrite(f"{save_dir}/{name}_bg_interactivemap_point{j+1}.png", bg_points_binary_map*255)


    # image = np.array(Image.open('/home/guiyan/workspaces/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg'))
    # # interactive_map = np.array(Image.open('./interactives/2007_000033_bg_interactivemap_object1_point20.png'))
    # interactive_map = np.array(Image.open('./interactives/2007_000033_fg_interactivemap_object1_point20.png'))

    # plt.axis('off')
    # plt.imshow(image)
    # y, x = np.where(interactive_map == 255)
    # plt.scatter(x, y, c='r')
    # # plt.show()
    # plt.savefig("test.png")


