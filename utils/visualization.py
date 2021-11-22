
## https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/cnn_layer_visualization.py
## https://zhuanlan.zhihu.com/p/75054200

import os
import torch
import numpy as np
import torchvision
from PIL import Image
import cv2 as cv
from sklearn import random_projection
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class VisualNetworkUtils():
    def __init__(self, model, visual_layer_infos,
                 visual_tool='summary', summary_file_dir='./visual', 
                 save_file=True, only_save_file=False, save_file_dir='./result'):

        assert visual_tool in ['summary', 'plot']

        self.summary_writer = None
        self.model = model

        if save_file and not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)

        if only_save_file:
            if visual_tool == 'summary':
                if not os.path.exists(summary_file_dir):
                    os.mkdir(summary_file_dir)
                self.summary_writer = SummaryWriter(log_dir=summary_file_dir)

        for module_name, visual_method in visual_layer_infos.items():
            # print(module_name, visual_method)
            module = self.__get_module(model, module_name)
            VisualNetworkUtils.VisualSingalLayer(module, module_name, self.summary_writer, 
                                input_visual_method=visual_method[0], 
                                output_visual_method=visual_method[1], 
                                only_save_file=only_save_file)

    def visual_model_graph(self, *inputs):
        self.summary_writer.add_graph(self.model, inputs) # visual model
        self.summary_writer.flush()

    def end_visual(self):
        self.summary_writer.close()

    def get_summary_writer(self):
        return self.summary_writer

    def __get_module(self, model, module_name):
    
        name_list = module_name.split('.')
        module = getattr(model, name_list[0])
        for name in name_list[1:]:
            module = getattr(module, name)
        
        return module

    class VisualSingalLayer():
        def __init__(self, module, layer_name, visual_tool,
                     input_visual_method=None, output_visual_method='per_channel', show_image_per_row=10, 
                     save_file=True, only_save_file=False, save_file_dir='./result'):
            
            module.register_forward_hook(self.__forward_hook)
            self.summary_writer = None
            self.layer_name = layer_name
            self.input_visual_method = input_visual_method
            self.output_visual_method = output_visual_method
            self.show_image_per_row = show_image_per_row
            self.save_file = save_file
            self.only_save_file = only_save_file
            self.save_file_dir = save_file_dir
            self.current_id = 0

            if isinstance(visual_tool, SummaryWriter):
                self.summary_writer = visual_tool
        
        def __forward_hook(self, module, inputs, outputs):
            self.__visual(inputs, is_output=False)
            self.__visual(outputs, is_output=True)
        
        def visual_tensor(self, tensor, title, visual_method):
            if not self.only_save_file:
                if self.summary_writer is not None:
                    VisualUtils.visual_by_summary(title, visual_method, summary_writer=self.summary_writer)
                else:
                    VisualUtils.visual_by_plot(tensor, title, visual_method, self.show_image_per_row)
            
            if self.save_file:
                VisualUtils.save_image(tensor, title, visual_method, self.save_file_dir)

        def __visual(self, tensors, is_output):
            flag = 'output' if is_output else 'input'
            if is_output:
                visual_method = self.output_visual_method
            else:
                visual_method = self.input_visual_method

            title = f'{self.current_id}_{self.layer_name}_{flag}'
            self.current_id += 1
            if isinstance(tensors, tuple) or isinstance(tensors, list):
                for i, tensor in enumerate(tensors):
                    self.visual_tensor(tensor, title+f"_{i}", visual_method)
            elif isinstance(tensors, dict):
                for key, tensor in tensors.items():
                    self.visual_tensor(tensor, title+f"_{key}", visual_method)
            elif isinstance(tensors, torch.Tensor):
                self.visual_tensor(tensors, title, visual_method)

class VisualUtils():
    @staticmethod
    def visual_by_summary(tensor, title, visual_method, 
                          summary_writer=None):   
        # print('visual_by_summary')
        b, c = tensor.shape[:2]
        if visual_method == 'image':
            for i in range(b):
                summary_writer.add_image(f'{title}_b{i+1}_image', tensor[i])
        elif visual_method == 'per_channel':
            # for i in range(b):
                # for j in range(c):
                    # self.summary_writer.add_image(f'{title}_b{i+1}_image', tensor[i][j], dataformats='HW')
            summary_writer.add_images(f'{title}_pc', tensor) # .cpu().detach()
        elif visual_method == 'random_projection':
            times = 3
            for i in range(times):
                tensor = VisualUtils.random_project_to_n_channel(tensor, out_channel=3)
                summary_writer.add_images(f'{title}_rp/{i+1}', tensor) # .cpu().detach()
        else:
            return
    
    @staticmethod
    def visual_by_plot(tensor, title, visual_method, show_image_per_row=10):
        # print('visual_by_plot')         
        b, c = tensor.shape[:2]
        print(f'{title}, shape: {tensor.shape}, visual_method: {visual_method}')
        if visual_method == 'image':
            for i in range(b):
                plt.figure()
                plt.subplot(1, b, i+1)
                title_temp = f'{title}_b{i+1}_image'
                # print(title_temp)
                plt.title(title_temp)
                plt.imshow(tensor[i].permute(1, 2, 0).cpu().detach().numpy())
                plt.show()
        elif visual_method == 'per_channel':
            for i in range(b):
                plt.figure()
                # c = visual_max_channel if c > visual_max_channel else c
                row = (c // show_image_per_row) + 1
                col = show_image_per_row if c > show_image_per_row else c % show_image_per_row
                # print(c, row, col)
                for j in range(c):
                    plt.subplot(row, col, j+1)
                    title_temp = f'{title}_b{i+1}_c{j+1}'
                    # print(title_temp)
                    plt.title(title_temp)
                    plt.imshow(tensor[i][j].cpu().detach().numpy())
                plt.show()
        elif visual_method == 'random_projection':
            times = 3
            for i in range(b):
                plt.figure()
                for j in range(times):
                    # tensor = VisualTensorUtils.random_project_to_n_channel(tensor, out_channel=1)
                    # tensor = torch.cat([tensor, tensor, tensor], dim=1)
                    tensor = VisualUtils.random_project_to_n_channel(tensor, out_channel=3)
                    title_temp = f'{title}_b{i+1}_image_{j+1}'
                    plt.subplot(b, times, i*times+j+1)
                    # print(title_temp)
                    plt.title(title_temp)
                    plt.imshow(tensor[i].permute(1, 2, 0).cpu().detach().numpy())
                plt.show()
        else:
            return
    
    @staticmethod
    def save_image(tensor, title, visual_method, save_dir):
        b, c = tensor.shape[:2]
        image_dict = {}
        if visual_method == 'image':
            for i in range(b):
                title_temp = f'{title}_b{i+1}_image'
                image_dict.update({title_temp: tensor[i].permute(1, 2, 0).cpu().detach().numpy()})
        elif visual_method == 'per_channel':
            for i in range(b):
                for j in range(c):
                    title_temp = f'{title}_b{i+1}_c{j+1}'
                    image_dict.update({title_temp: tensor[i][j].cpu().detach().numpy()})
        elif visual_method == 'random_projection':
            times = 3
            for i in range(b):
                for j in range(times):
                    tensor = VisualUtils.random_project_to_n_channel(tensor, out_channel=3)
                    title_temp = f'{title}_b{i+1}_image_{j+1}'
                    image_dict.update({title_temp: tensor[i].permute(1, 2, 0).cpu().detach().numpy()})

        for key in image_dict.keys():
            # print(key)
            # image_dict[key] = (image_dict[key] - image_dict[key].min() )/ (image_dict[key].max() - image_dict[key].min())
            plt.imsave(os.path.join(save_dir, f'{key}.jpg'), image_dict[key], cmap='viridis')

    def show_images(*images, n_row=1, n_col=None, show_axis=False):
        if n_col is None:
            n_col = len(images)
        plt.figure(figsize=(4 * n_col,4 * n_row))
        for i, img in enumerate(images):

            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, torch.Tensor):
                img = torchvision.transforms.ToPILImage()(img)
            
            if isinstance(img, Image.Image) is not True:
                raise Exception('only support [numpy.ndarray, torch.Tensor, PIL.Image]')
            
            plt.subplot(n_row, n_col, i+1)
            plt.axis('on' if show_axis else 'off')
            plt.imshow(img)
        plt.show()


    @staticmethod
    def random_project_to_n_channel(embedding, out_channel=3):
        transformer = random_projection.GaussianRandomProjection(n_components=out_channel)

        b, c, h, w = embedding.shape
        device = embedding.device

        embeddings = []
        for i in range(b):
            embedding_temp = embedding[i].permute(1, 2, 0) #  (c, h, w) -> (h, w, c)
            embedding_temp = embedding_temp.reshape(h * w, c) # (h, w, c) -> (h * w, c)

            embedding_temp = embedding_temp.cpu()
            embedding_temp = transformer.fit_transform(embedding_temp.detach().numpy()) # (h * w, c) -> (h * w, 3)
            embedding_temp = embedding_temp.reshape(h, w, out_channel)  # (h * w, 3) -> (h, w, 3)
            
            embedding_temp = torch.from_numpy(embedding_temp).to(device)
            embedding_temp = embedding_temp.permute(2, 0, 1).unsqueeze(0) #  (h, w, 3) -> (1, 3, h, w)
            embeddings.append(embedding_temp)

        embeddings = torch.cat(embeddings)

        embeddings = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())

        return embeddings



# model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
# # print(model)
# visual_layer_infos = {
#     # layer_name: [
#         # visual_method: per_channel or random_projection or image
#     # ]
#     'backbone': ['image', 'random_projection'],
#     'classifier': ['None', 'random_projection']
# }
# # VisualizationUtils(model, visual_layer_infos, visual_tool='plot')
# model.eval()
# # model.cuda()
# # print(model)
# x = torch.rand((2, 3, 100, 100))
# VisualNetworkUtils(model, visual_layer_infos, visual_tool='plot', only_save_file=True)
# out = model.forward(x)
# x = torch.rand((1, 3, 100, 100))
# out = model.forward(x)