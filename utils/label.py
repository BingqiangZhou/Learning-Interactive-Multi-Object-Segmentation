import random
import torch

class LabelUtils():
    
    def get_nums_label(dataset_name, label, p=0.5, shuffle=True):
        if dataset_name == 'VOC':
            if label.max() == 1:
                label = label * 255
            label[label == 255] = 0
            label = LabelUtils.update_label(label, p, shuffle=shuffle)
        return label
    
    @staticmethod
    def update_label(label, p=0.5, shuffle=True):
        label_ids = list(label.unique().numpy())
        temp_label = torch.zeros_like(label) # (1, 1, h, w)
        index = 1
        if shuffle:
            random.shuffle(label_ids)
        for id in label_ids:
            if id == 0 or id == 255:
                continue
            if shuffle and index > 2 and random.random() < p: # when the number of object more than 2, random transform fg to bg
                continue
            else:
                temp_label[label == id] = index
                index += 1
        # print(index, temp_label.unique())
        return temp_label

    # @staticmethod
    # def update_label(label, p=0.5, shuffle=True):
    #     label_ids = label.unique()
    #     index = 1
    #     if shuffle:
    #         random.shuffle(label_ids)
    #     for i, id in enumerate(label_ids):
    #         if i > 1 and random.random() < p: # when the number of object more than 2, random transform fg to bg
    #             label[label == id] = 0
    #         else:
    #             label[label == id] = index
    #             index += 1
        
    #     return label


    # @staticmethod
    # def update_label(label):
    #     label_ids = label.unique()
    #     for i, id in enumerate(label_ids):
    #         label[label == id] = i
        
    #     return label