import time
import torch
import numpy as np
import torchvision.transforms.functional as transfrom_funcs

from config import Config
from builder import Builder

class OurNet():
    def __init__(self, config, model_path) -> None:
        basic_config = config.get('BASIC')
        network_configs = config.get('NETWORK')
        self.device = torch.device('cpu')
        if basic_config.get('USE_GPU'):
            gpu_index = basic_config.get('GPU')
            self.device = torch.device(f'cuda:{gpu_index}')
        self.model = Builder.build_model(basic_config, network_configs, self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def __normalized(self, image_tensor):
        image_tensor_normalized = transfrom_funcs.normalize(image_tensor.float(), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return image_tensor_normalized


    def predict_from_numpy(self, image:np.ndarray, interactive_distance_map:np.ndarray, normalized=False):
        '''
            image: (h, w, 3),
            interactive_distance_map: (h, w, n)

            return:
                outputs: (h, w, n+1)
        '''
        image_tensor = torch.from_numpy(np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)) # (h, w, 3) -> (1, 3, h, w)
        interactive_map_tensor = torch.from_numpy(np.expand_dims(np.transpose(interactive_distance_map, (2, 0, 1)), axis=0)) # (h, w, n) -> (1, n, h, w)
        
        outputs_tensor, spend_times = self.predict_from_tensor(image_tensor, interactive_map_tensor, normalized)

        output_np = np.transpose(outputs_tensor.detach().cpu().squeeze(dim=0).numpy(), (1, 2, 0)) # (1, n+1, h, w) -> (h, w, n+1)
        return output_np, spend_times
    
    def predict_from_tensor(self, image_tensor:torch.Tensor, interactive_distance_map_tensor:torch.Tensor, normalized=False):
        '''
            image: (1, 3, h, w),
            interactive_distance_map: (1, n, h, w)

            return:
                outputs: (1, n+1, h, w)
        '''
        image_tensor = image_tensor / 255.0
        # interactive_map_tensor = interactive_distance_map_tensor / 255.0
        if normalized is False:
            image_tensor = self.__normalized(image_tensor)
        
        image_tensor = image_tensor.to(self.device)
        interactive_map_tensor = interactive_distance_map_tensor.to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            outputs_tensor, embedding, visual_attention_maps, margin = self.model(image_tensor, interactive_map_tensor.float())
            end_time = time.time()
        return outputs_tensor, end_time - start_time

