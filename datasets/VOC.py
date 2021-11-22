

import os
import numpy as np
from PIL import Image
import torchvision

class VOCSegmentation():
    '''
        VOC dataset class for Instance Segmentation.
    '''
    def __init__(self, root_dir, target_type='Object', image_set='train', transforms=None):
        super(VOCSegmentation, self).__init__()

        assert target_type in ['Object', 'Class'], "`target_type` must in ['Object', 'Class']"
        assert image_set in ['train', 'val', 'trainval'], "`image_set` in ['train', 'val', 'trainval']"

        self.root = root_dir
        self.transforms = transforms

        base_dir = 'VOCdevkit/VOC2012'
        voc_root_dir = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root_dir, 'JPEGImages')
        mask_dir = os.path.join(voc_root_dir, 'Segmentation'+target_type)

        if not os.path.isdir(voc_root_dir):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root_dir, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            image = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)

        return image, target

    def __len__(self):
        return len(self.images)