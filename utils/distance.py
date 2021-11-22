

import torch
import numpy as np
from scipy import ndimage
from scipy.spatial import distance

class DistanceUtils():
    '''
    calulate distance map for user's interactive,
    user's interactive is binary map, the value of user's interactive position is 1
    '''
    @staticmethod
    def bwdist(binary_mask):
        '''
        the distance of background element(0) to the closest foreground element(1)
        '''
        is_torch_tensor = False
        if isinstance(binary_mask, torch.Tensor):
            device = binary_mask.device
            is_torch_tensor = True
            binary_mask = binary_mask.cpu().numpy()
        distance_map = ndimage.morphology.distance_transform_edt(1 - binary_mask)   
        if is_torch_tensor:
            distance_map = torch.from_numpy(distance_map).float().to(device)
        return distance_map

    @staticmethod
    def get_euclidean_distance_map(binary_masks):
        '''
        Description:
            euclidean distance map
        Arguments: 
            binary_masks (torch.Tensor): binary masks (1, n, h, w), in our network, this input is random sample point map.
        Return: 
            euclidean_distance_map (torch.Tensor): binary masks (1, n, h, w)
        '''
        b, c, h, w = binary_masks.shape

        assert binary_masks.dim() == 4, r"binary_mask's shape must be (b, n, h, w)"
        
        device = binary_masks.device
        tensor_on_device = device.__str__()[:4]
        if tensor_on_device == 'cuda':
            binary_masks = binary_masks.cpu()

        batch = []
        for i in range(b):
            euclidean_distance_map = []
            for j in range(c):
                distance_map = DistanceUtils.bwdist(binary_masks[i][j].numpy())
                distance_map = torch.from_numpy(distance_map).type(torch.float32)
                euclidean_distance_map.append(distance_map)
            euclidean_distance_map = torch.stack(euclidean_distance_map, dim=0) # (n, h, w)
            euclidean_distance_map[euclidean_distance_map > 255] = 255
            # euclidean_distance_map = euclidean_distance_map / 255
            batch.append(euclidean_distance_map)
        batch = torch.stack(batch, dim=0)

        batch = batch.to(device)

        return batch
    
    @staticmethod
    def get_euclidean_distance_map_from_embedding(embedding, object_select_point_binary_map):
        b, c, h, w = object_select_point_binary_map.shape
        batch = []
        for i in range(b):
            euclidean_distance_map = []
            for j in range(c):
                distance_map = DistanceUtils.get_euclidean_distance_map_from_embedding_one_object(embedding[i], object_select_point_binary_map[i][j])
                euclidean_distance_map.append(distance_map)
            euclidean_distance_map = torch.stack(euclidean_distance_map, dim=0) # (n, h, w)
            # euclidean_distance_map[euclidean_distance_map > 255] = 255
            batch.append(euclidean_distance_map)
        batch = torch.stack(batch, dim=0)
        return batch

    @staticmethod
    def get_euclidean_distance_map_from_embedding_one_object(embedding, one_object_select_point_binary_map):
        '''
        @Description: 
        @Arguments: 
            embedding: (n, h, w)
            one_object_select_point_binary_map: (h, w), 0 is background
        @Return: 
        '''
        # (n)
        # print(torch.sum(one_object_select_point_binary_map * embedding, dim=(1, 2)))
        select_center_vector = torch.sum(one_object_select_point_binary_map * embedding, dim=(1, 2)) / torch.sum(one_object_select_point_binary_map)

        embedding = embedding.permute(1, 2, 0) # (n, h, w) -> (h, w, n)
        h, w, n = embedding.size() # 
        embedding = embedding.reshape(h * w, n) # (h, w, n) -> (h*w, n)
        
        distance_map = torch.sum((embedding - select_center_vector)**2, 1) # (h*w)

        distance_map = distance_map.reshape(h, w)
        
        return distance_map

 

    