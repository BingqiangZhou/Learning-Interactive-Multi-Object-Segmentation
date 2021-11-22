

import torch
import torchvision
import random
import time

class Transfroms(torchvision.transforms.Compose):
    def __call__(self, *args):
        img_list = []
        img_list.extend(args)
        for transform in self.transforms:
            seed_time = int(round(time.time() * 1000)) # ms
            for i in range(len(img_list)):
                #Since v0.8.0 all random transformations are using torch default random generator to sample random parameters. 
                if self.get_version_integer() > 80:
                    torch.manual_seed(seed_time)
                else:
                    random.seed(seed_time)
                img_list[i] = transform(img_list[i])
        return img_list
    
    def get_version_integer(self):
        version_str = torchvision.__version__.split('+')[0] # '0.9.1+cu111'(pip)，'0.9.1'(conda)
        version_int = int(version_str.replace('.', ''))
        return version_int
