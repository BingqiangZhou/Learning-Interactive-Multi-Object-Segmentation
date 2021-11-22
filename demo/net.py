import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.getcwd(), os.path.pardir))

from config import Config
from builder import Builder
from utils import DistanceUtils

class MOSNet():
    def __init__(self, device_num=0, config_file_path='../exp_results/t.yaml', 
                 model_file_path='../model/best_mean_iou_epoch.pkl'):
        
        config = Config(config_file_path).get_configs()
        basic_config = config.get('BASIC')
        network_configs = config.get('NETWORK')
        device_str = "cpu" if device_num < 0 else f'cuda:{device_num}'
        self.device = torch.device(device_str)
        self.model = Builder.build_model(basic_config, network_configs, self.device)
        state_dict = torch.load(model_file_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def points_list_to_binary_interactive_map(self, points_list, size):
        '''
            points_list: ndarray, [n, 3], like [[x, y, i], [x, y, i], ...] 
        '''
        binary_interactive_map = []
        for i in range(points_list.shape[0]):
            x, y, object_nums = points_list[i]
            if len(binary_interactive_map) <= object_nums-1:
                interactive_map = np.zeros(size)
                interactive_map[int(y), int(x)] = 1
                binary_interactive_map.append(interactive_map)
            else:
                binary_interactive_map[object_nums-1][y, x] = 1
        binary_interactive_map = np.stack(binary_interactive_map, axis=0) # (c, h, w)
        return torch.from_numpy(binary_interactive_map[None, :, :, :])  # (1, c, h, w)
    
    def predict(self, image, interactive_points, extra_output=True):
        
        size = image.shape[:2]
        image_tensor = self.transfrom(image).unsqueeze(0).to(self.device) # (3, h, w) --> (1, 3, h, w)
        binary_interactive_map = self.points_list_to_binary_interactive_map(interactive_points, size)
        # ## transfrom binary interactive map to distance map
        distance_map = DistanceUtils.get_euclidean_distance_map(binary_interactive_map).to(self.device)
        with torch.no_grad():
            outputs, embedding, visual_attention_maps, margin = self.model(image_tensor, distance_map) 
        outputs_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy() #(1, n+1, h, w) -> #(h, w)
        if extra_output:
            embedding_np = embedding[0].cpu().numpy()# (1, 128, h ,w) -> (128, h, w)
            visual_attention_maps_np = visual_attention_maps[0].cpu().numpy() # (1, n , h, w) -> (n, h, w)
            return outputs_mask, embedding_np, visual_attention_maps_np
        return outputs_mask

# points_list = [(120, 200, 1), (400, 200, 2)]
# points_list = [(120, 200, 1)]
# net = MOSNet()
# img = np.array(Image.open('./2007_000925.jpg'))
# out, embedding_np, visual_attention_maps_np = net.predict(img, points_list)
# print(out.shape, np.unique(out))
# plt.imshow(out/255)
# print("done!")