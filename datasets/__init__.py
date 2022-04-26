
from .VOC import VOCSegmentation
# from .COCO import COCOSegmeation
# from .Cityscapes import CityscapesSegmenation
# from .DAVIS import DAVISSegmenation
from .transforms import Transfroms

class Datasets():
    def __init__(self, name):

        assert name in ['VOC', 'COCO', 'Cityscapes', 'DAVIS'], \
                'only support dataset: [VOC, COCO, Cityscapes, DAVIS]'

        self.name = name

    def get_dataset(self, *args, **kwargs):
        if self.name == 'VOC':
            # root_dir, target_type='Object', image_set='train', transforms=None
            dataset = VOCSegmentation(*args, **kwargs)
        # elif self.name == 'COCO':
        #     # root_dir, image_set='train', year='2017', transforms_dict=None
        #     dataset = COCOSegmeation(*args, **kwargs)
        # elif self.name == 'Cityscapes':
        #     # root_dir, image_set='train', mode='fine', transforms_dict=None
        #     dataset = CityscapesSegmenation(*args, **kwargs)
        # else:  
        #     '''DAVIS'''
        #     # root_dir, image_set='train', year='2017', resolution='480p', transforms_dict=None
        #     dataset = DAVISSegmenation(*args, **kwargs)
        return dataset