from numpy import random
import torch
import numpy as np
from PIL import Image

from .distance import DistanceUtils

bwdist = DistanceUtils.bwdist

class RandomSamplePointUtils():
    
    @staticmethod
    def sample_point_from_deep_selection(binary_mask, num_points=10, d_step=10, d_margin=5):
        points_binary_map = np.zeros_like(binary_mask) # positive channel
        original_sample_region = np.int8((bwdist(1-binary_mask) > d_margin))
        for i in range(num_points):
            if i == 0:
                sample_region = original_sample_region
            else: #  i > 0
                sample_region = np.int8(original_sample_region + (bwdist(points_binary_map) > d_step) == 2)
            sample_map = np.random.rand(*(sample_region.shape)) * sample_region
            max_value = np.max(sample_map)
            if max_value == 0:
                break
            else:
                max_index = np.argmax(sample_map)
                # x = index / ncols , y = index % ncols
                # points_binary_map[index // binary_mask.shape[1], index % binary_mask.shape[1]] = 1
                x, y = np.unravel_index(max_index, sample_map.shape)
                points_binary_map[x, y] = 1
        return points_binary_map

    @staticmethod
    def sample_point_by_adaptive_dmargin(binary_mask, num_points=10, dmargin_interval=[0.05, 0.5]):
        points_binary_map = np.zeros_like(binary_mask) # positive channel
        max_margin = np.max(bwdist(1-binary_mask))
        for i in range(num_points):
            d_margin = np.random.randint(max_margin * dmargin_interval[0], max_margin * dmargin_interval[1] + 1)
            # print(d_margin)
            sample_region = bwdist(1-binary_mask) > d_margin
            if i == 0:
                sample_map = np.random.rand(*(sample_region.shape)) * sample_region
                max_index = np.argmax(sample_map)
            else:
                max_index = np.argmax(bwdist(points_binary_map) * sample_region)
            # points_binary_map[index // binary_mask.shape[1], index % binary_mask.shape[1]] = 1
            x, y = np.unravel_index(max_index, sample_map.shape)
            points_binary_map[x, y] = 1
        return points_binary_map

    @staticmethod
    def sample_point_by_adaptive_dstep(binary_mask, num_points=10, margin_interval=[0.5, 0.8]):
        points_binary_map = np.zeros_like(binary_mask) # positive channel
        sample_region = binary_mask
        max_d_step = np.max(bwdist(1-binary_mask))
        for i in range(num_points):
            d_step = np.random.randint(max_d_step * margin_interval[0], max_d_step * margin_interval[1] + 1)
            sample_map = np.random.rand(*(sample_region.shape)) * sample_region
            max_value = np.max(sample_map)
            if max_value == 0:
                break
            else:
                max_index = np.argmax(sample_map)
                # x = index / ncols , y = index % ncols
                # points_binary_map[index // sample_map.shape[1], index % sample_map.shape[1]] = 1
                x, y = np.unravel_index(max_index, sample_map.shape)
                points_binary_map[x, y] = 1
                temp_points_binary_map = np.zeros_like(binary_mask)
                # temp_points_binary_map[index // sample_map.shape[1], index % sample_map.shape[1]] = 1
                temp_points_binary_map[x, y] = 1
                sample_region = np.int8(sample_region + (bwdist(temp_points_binary_map) > d_step) == 2)
        return points_binary_map

    @staticmethod
    def sample_background_points(label_binary_mask, num_points=10, d_step=10, d_margin=5, d=40):
        fg = 1 - label_binary_mask
        sample_region = np.int8((bwdist(fg) < d) - fg)
        return RandomSamplePointUtils.sample_point_from_deep_selection(sample_region, num_points, d_step, d_margin)

    @staticmethod
    def sample_points_from_label2d(label, num_points=10, sample_method="random_sample", d_step=10, d_margin=5,
                        dmargin_interval=[0.05, 0.5], dstep_interval=[0.5, 0.8], sample_background_points=False):

        assert sample_method in ["random_sample", "adaptive_dmargin", "adaptive_dstep"]
        assert len(label.shape) == 2

        is_tensor = isinstance(label, torch.Tensor)
        if is_tensor:
            label_np = label.cpu().numpy()
            device = label.device
        else:
            label_np = np.array(label)
        points_binary_maps = []
        # print(np.unique(label_np))
        # assert len(np.unique(label_np)) > 1
        for label_num in np.unique(label_np):
            label_binary_mask = (label_np == label_num).astype(np.uint8)
            if label_num == 0:
                if sample_background_points: # background
                    background_points_binary_map = RandomSamplePointUtils.sample_background_points(label_binary_mask, num_points * 2, 
                                                                                                    d_step, d_margin)
                    points_binary_maps.append(background_points_binary_map)
                continue
            # label_binary_mask = np.zeros_like(label_np)
            # label_binary_mask[label_np == label_num] = 1
            if sample_method == "adaptive_dmargin":
                # print("adaptive_dmargin")
                points_binary_map = RandomSamplePointUtils.sample_point_by_adaptive_dmargin(label_binary_mask, num_points, dmargin_interval)
            elif sample_method == "adaptive_dstep":
                # print("adaptive_dstep")
                points_binary_map = RandomSamplePointUtils.sample_point_by_adaptive_dstep(label_binary_mask, num_points, dstep_interval)
            else: # sample_method == "random_sample"
                # print("random_sample")
                points_binary_map = RandomSamplePointUtils.sample_point_from_deep_selection(label_binary_mask, num_points, d_step, d_margin)
            # points_binary_map = bwdist(points_binary_map)
            points_binary_maps.append(points_binary_map)
        points_binary_maps = np.stack(points_binary_maps, axis=0) # (c, h, w)
        if is_tensor:
            points_binary_maps = torch.from_numpy(points_binary_maps).to(device)
        return points_binary_maps
    
    @staticmethod
    def sample_points_from_label_tensor(label_tensor, num_points=10, sample_method="random_sample", d_step=10, d_margin=5,
                        dmargin_interval=[0.05, 0.5], dstep_interval=[0.5, 0.8], sample_background_points=False):
        # label_tensor (n, 1, h, w) -> (n, c, h, w)
        n, _, h, w = label_tensor.shape
        points_binary_maps = []
        for i in range(n):
            points_binary_maps.append(RandomSamplePointUtils.sample_points_from_label2d(label_tensor[i][0], num_points, sample_method, 
                                        d_step, d_margin, dmargin_interval, dstep_interval, sample_background_points))
        points_binary_maps = torch.stack(points_binary_maps, dim=0) # (n, c, h, w)
        return points_binary_maps
    
    @staticmethod
    def sample_points(label_tensor, num_points=10, 
                        sample_method_list=["random_sample", "adaptive_dstep", "adaptive_dmargin"], 
                        d_step=10, d_margin=5,
                        dmargin_interval=[0.05, 0.5], dstep_interval=[0.5, 0.8],
                        sample_background_points=False, random_nums_points=False):
        sample_method = random.choice(sample_method_list)
        if random_nums_points:
            num_points = random.randint(1, num_points+1)
        return RandomSamplePointUtils.sample_points_from_label_tensor(label_tensor, num_points, 
                                        sample_method, d_step, d_margin,
                                        dmargin_interval, dstep_interval, sample_background_points), num_points

# class RandomSamplePointUtils():
#     @staticmethod
#     def sample_point_from_paper(binary_mask, num_points=10, d_step=10, d_margin=5):
#         '''
#         Random sampling points instead of interaction, 
#         and sampling strategy from Deep Interactive Object Selection
#         https://arxiv.org/abs/1603.04042
#         '''
        
#         device = binary_mask.device
#         points_binary_map = torch.zeros_like(binary_mask) # positive channel
#         for i in range(num_points):
#             sample_region = (bwdist(1-binary_mask) > d_margin)
#             if i > 0:
#                 sample_region = (sample_region + (bwdist(points_binary_map) > d_step) == 2)
#             sample_map = torch.rand(*(sample_region.shape), device=device) * sample_region
#             max_value = torch.max(sample_map)
#             if max_value == 0:
#                 break
#             else:
#                 index = torch.argmax(sample_map)
#                 # x = index / ncols , y = index % ncols
#                 points_binary_map[index // binary_mask.shape[1], index % binary_mask.shape[1]] = 1
#         return points_binary_map

#     def sample_point_strategy_1(binary_mask, num_points=10, margin_interval=[0.05, 0.5]):
#         device = binary_mask.device
#         points_binary_map = torch.zeros(*(binary_mask.shape), device=device) # positive channel
#         max_margin = torch.max(bwdist(1-binary_mask)).item()
#         for i in range(num_points):
#             d_margin = random.randint(int(max_margin * margin_interval[0]),
#                                      int(max_margin * margin_interval[1]) + 1)
#             sample_region = (bwdist(1-binary_mask) > d_margin).float()
#             if i == 0:
#                 sample_map = torch.rand(*(sample_region.shape), device=device) * sample_region
#                 # print(sample_map.shape)
#                 index = torch.argmax(sample_map)
#             else:
#                 index = torch.argmax(bwdist(points_binary_map) * sample_region)
#             points_binary_map[index // binary_mask.shape[1], index % binary_mask.shape[1]] = 1
#         return points_binary_map

#     @staticmethod
#     def sample_point_strategy_2(binary_mask, num_points=10, margin_interval=[0.5, 0.8]):
#         device = binary_mask.device
#         points_binary_map = torch.zeros(*(binary_mask.shape), device=device) # positive channel
#         sample_region = binary_mask.float()
#         max_d_step = torch.max(bwdist(1-binary_mask)).item()
#         for i in range(num_points):
#             d_step = random.randint(int(max_d_step * margin_interval[0]), 
#                                 int(max_d_step * margin_interval[1]) + 1)
#             # print('d_step', d_step)
#             sample_map = torch.rand(*(sample_region.shape), device=device) * sample_region
#             max_value = torch.max(sample_map)
#             if max_value == 0:
#                 # print('break')
#                 break
#             else:
#                 index = torch.argmax(sample_map)
#                 # x = index / ncols , y = index % ncols
#                 x = index // sample_map.shape[1]
#                 y = index % sample_map.shape[1]
#                 points_binary_map[x, y] = 1
#                 temp_points_binary_map = torch.zeros(*(binary_mask.shape), device=device)
#                 temp_points_binary_map[x, y] = 1
#                 sample_region = ((sample_region + (bwdist(temp_points_binary_map) > d_step).float()) == 2).float()
#                 # plt.figure()
#                 # img = torchvision.transforms.ToPILImage()(sample_region)
#                 # plt.imshow(img)
#         return points_binary_map

#     @staticmethod
#     def sample_points_by_random_strategy(label_tensor, num_points=10, p=0.5,
#                                         strategy_1_margin_interval=[0.05, 0.5],
#                                         strategy_2_margin_interval=[0.5, 0.8]):
#         device = label_tensor.device
#         b, _, h, w = label_tensor.shape # (b, 1, h, w)
#         points_binary_maps_list = []
#         for i in range(b):
#             points_binary_maps = []
#             current_label = label_tensor[i][0]
#             label_nums = torch.unique(current_label)
#             if len(label_nums) <= 1: # return None when no foreground
#                 return None
#             for label_num in label_nums:
#                 if label_num == 0: # skip background
#                     continue
#                 label_binary_mask = torch.zeros(*(current_label.shape), device=device)
#                 label_binary_mask[current_label == label_num] = 1
#                 if random.random() > p: # strategy 1:p, strategy 2: 1-p
#                     # print('strategy 1')
#                     points_binary_map = RandomSamplePointUtils.sample_point_strategy_1(label_binary_mask, num_points, strategy_1_margin_interval)
#                 else:
#                     # print('strategy 2')
#                     points_binary_map = RandomSamplePointUtils.sample_point_strategy_2(label_binary_mask, num_points, strategy_2_margin_interval)
#                 # points_binary_map = bwdist(points_binary_map)
#                 points_binary_maps.append(points_binary_map)
#             points_binary_maps = torch.stack(points_binary_maps, dim=0).to(device) # (c, h, w)
#             points_binary_maps_list.append(points_binary_maps)
#         points_binary_maps_tensor = torch.stack(points_binary_maps_list, dim=0).to(device) #(b, c, h, w)
#         return points_binary_maps_tensor