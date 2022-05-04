import os
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
voc_root_dir = '/home/guiyan/workspaces/datasets/voc2012/VOCdevkit/VOC2012'
mask_dir = os.path.join(voc_root_dir, 'SegmentationObject')
splits_dir = os.path.join(voc_root_dir, 'ImageSets/Segmentation')
split_f = os.path.join(splits_dir, 'val.txt')

# ----- random sample parameters -----
max_num_points = 20
dmargin_interval=[0.05, 0.5]
d = 40
margin = 5
step = 10

# ----- the dir of saving interactive map -----
save_dir = "./interactives"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ----- get all file name from val dataset -----
with open(os.path.join(split_f), "r") as f:
    file_names = [x.strip() for x in f.readlines()]

nums_file = len(file_names)

for index, name in enumerate(file_names):
    file_path = os.path.join(mask_dir, f"{name}.png")
    label = np.array(Image.open(file_path))
    label[label == 255] = 0 # set the value of object's contours to 0(background) 
    ids = np.unique(label)
    print(f"{index+1} / {nums_file} - {name} : {ids}")
    for i, id in enumerate(ids):
        if id == 0:
            continue
        binary_mask = np.uint8(label == id)
        
        fg_points_binary_map = np.zeros_like(binary_mask) # positive channel
        bg_points_binary_map = np.zeros_like(binary_mask) # negative channel
        
        max_margin = np.max(bwdist(1-binary_mask))
        
        bg_sample_region = np.int8((bwdist(binary_mask) < d) - binary_mask)
        original_bg_sample_region = np.int8((bwdist(1-bg_sample_region) > margin))

        for j in tqdm(range(max_num_points), desc=f'{name}-{id}'):
            
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
            cv.imwrite(f"{save_dir}/{name}_fg_interactivemap_object{i}_point{j+1}.png", fg_points_binary_map*255)

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
            cv.imwrite(f"{save_dir}/{name}_bg_interactivemap_object{i}_point{j+1}.png", bg_points_binary_map*255)


# image = np.array(Image.open('/home/guiyan/workspaces/datasets/voc2012/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg'))
# # interactive_map = np.array(Image.open('./interactives/2007_000033_bg_interactivemap_object1_point20.png'))
# interactive_map = np.array(Image.open('./interactives/2007_000033_fg_interactivemap_object1_point20.png'))

# plt.axis('off')
# plt.imshow(image)
# y, x = np.where(interactive_map == 255)
# plt.scatter(x, y, c='r')
# # plt.show()
# plt.savefig("test.png")


