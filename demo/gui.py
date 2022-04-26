from args import Args
from codes.utils import random_project_to_n_channel

import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import numpy as np
import cv2 as cv
import os

def main_window():
    Args.cur_fig = plt.figure("Multi-Object Segmentation", figsize=(8, 6))
    Args.cur_fig.canvas.mpl_connect("button_press_event", mouse_press)
    Args.cur_fig.canvas.mpl_connect("key_press_event", key_press)
    show_title()
    plt.imshow(Args.show_image)        
    color_map_bar()
    plt.axis("off")
    plt.show()

def color_map_bar():
    # https://matplotlib.org/stable/tutorials/colors/colorbar_only.html#sphx-glr-tutorials-colors-colorbar-only-py
    cmap = mpl.cm.Set1
    norm = mpl.colors.Normalize(vmin=0, vmax=cmap.N)

    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
                orientation='horizontal',
                ticks=[0.5+i for i in range(cmap.N)],
                label="color map for interact point and result mask")
    cb.set_ticklabels([str(i+1) for i in range(cmap.N)])
    # plt.axis("off")
    # plt.show()

def show_title():
    mode_str = "auto predict" if Args.auto_predict else "press 'p' to predict"
    plt.title(f"[{mode_str}] current interactive {Args.cur_object_index}-th object")
    # Args.cur_fig.canvas.draw()
    # if Args.cur_fig.canvas is not None:
    #     Args.cur_fig.canvas.refresh()
def init():
    Args.points_list.clear()
    Args.result = None
    Args.cur_object_index = 1
    

def refresh():
    plt.cla()
    image_path = Args.image_paths[Args.cur_image_index]
    Args.cur_image_name = os.path.splitext(os.path.split(image_path)[1])[0]
    Args.source_image = np.array(Image.open(image_path))
    Args.show_image = Args.source_image.copy()
    plt.imshow(Args.show_image)
    if Args.result is not None:
        show_result = np.zeros_like(Args.show_image)
        for i in np.unique(Args.result):
            if i == 0:
                continue
            else:
                show_result[Args.result == i] = Args.color_map[i-1]
        plt.imshow(show_result, alpha=0.5)
    plt.axis("off")
    Args.cur_object_index = 1 if Args.points_list.number_of_array == 0 else Args.points_list.data[-1][-1]
    for k in range(Args.points_list.number_of_array):
        x, y, i = Args.points_list.data[k]
        plt.scatter(x, y, c=[Args.color_map[i-1]/255])
        plt.scatter(x, y, c='', marker='o', edgecolors='white')
    show_title()
    Args.cur_fig.canvas.draw()

def save(name, save_dir='./outputs'):
    now_str = datetime.now().__str__().replace(' ', '_')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    out_dir = os.path.join(save_dir, f'{name}_{now_str}')
    os.mkdir(out_dir)
    if Args.result is not None:
        plt.imsave(os.path.join(out_dir, 'outputs.png'), Args.result/255, cmap="gray")
#         plt.imsave(os.path.join(save_dir, f'{name}_interactives_{now_str}.png'), show_image)
        plt.title("")
        plt.savefig(os.path.join(out_dir, f'outputs_with_interactives.png'))
    if Args.embedding is not None:
        for i in range(3):
            embedding_np = random_project_to_n_channel(Args.embedding)
            plt.imsave(os.path.join(out_dir, f'embedding_{i+1}.png'), embedding_np)
    if Args.visual_attention_maps is not None:
        for i in range(Args.visual_attention_maps.shape[0]):
            plt.imsave(os.path.join(out_dir, f'visual_attention_{i+1}.png'), Args.visual_attention_maps[i], cmap="gray")
    if Args.points_list.number_of_array > 0:
        interactive_image = Args.source_image.copy()
        np.save(os.path.join(out_dir, f'interactives.npy'), Args.points_list.data)
        for k in range(Args.points_list.number_of_array):
            x, y, i = Args.points_list.data[k]
            cv.circle(interactive_image, (x, y), 10, Args.color_map[i-1].tolist(), -1)
            cv.circle(interactive_image, (x, y), 10, (255, 255, 255), 2)
        cv.imwrite(os.path.join(out_dir, f'interactives.png'), interactive_image[:,:,::-1])

def mouse_press(event):
    cur_points_num = Args.points_list.number_of_array
    if event.button == 1:  # 点击鼠标左键 进行交互
        plt.scatter(event.xdata, event.ydata, c=[Args.color_map[Args.cur_object_index-1]/255])
        plt.scatter(event.xdata, event.ydata, c='', marker='o', edgecolors='white')
        Args.points_list.push((int(event.xdata), int(event.ydata), Args.cur_object_index))
#         print("x,y=", event.xdata, event.ydata)
        Args.cur_fig.canvas.draw()
    elif event.button == 3:  # 点击鼠标右键 取消上次交互信息
        if Args.points_list.number_of_array - 1 > 0: 
            Args.points_list.pop()
            Args.cur_object_index = Args.points_list.data[-1, 2]
            refresh()
        elif Args.points_list.number_of_array - 1 == 0:
            Args.points_list.pop()
            Args.cur_object_index = 1
            Args.result = None
            refresh()
    
    if Args.auto_predict and cur_points_num != Args.points_list.number_of_array:
        if Args.points_list.number_of_array > 0:
            plt.title(f"predicting...")
            Args.result, Args.embedding, Args.visual_attention_maps = Args.net.predict(Args.source_image, Args.points_list.data)
            refresh()
        else:
            Args.result = None
            refresh()

def key_press(event):
    if event.key in Args.object_nums_str:
        if Args.points_list.number_of_array == 0:
            Args.cur_object_index = 1
        else:
            interacted_object_num = Args.points_list.data[:, 2].max()
            if int(event.key) > interacted_object_num:
                Args.cur_object_index = interacted_object_num + 1
            else:
                Args.cur_object_index = int(event.key)
        show_title()
    elif event.key == 'p' and not Args.auto_predict: # 预测
        plt.title(f"predicting...")
        Args.result, _, _ = Args.net.predict(Args.source_image, Args.points_list.data)
        refresh()
    elif event.key == 'ctrl+alt+s': # 保存
        plt.title(f"saving...")
        save(Args.cur_image_name, save_dir='./outputs')
        show_title()
        refresh()
    elif event.key == 'c':
        Args.auto_predict = not Args.auto_predict
        show_title()
    elif event.key == 'b':
        Args.cur_image_index = Args.max_image_index - 1 if Args.cur_image_index - 1 < 0 else Args.cur_image_index - 1
        init()
        refresh()
    elif event.key == 'a':
        Args.cur_image_index = Args.cur_image_index + 1 if Args.cur_image_index + 1 < Args.max_image_index else 0
        init()
        refresh()
    elif event.key == 'r':
        init()
        refresh()


readme_string = """
    operation: 
        mouse: 
            [left button]：interacte
            [right button]：cancel last interactation
        keyboard:
            [number key, include 1-9]: n-th object mark
            ['p' key]: predict result when not in "auto predict" mode
            ['k' key]：save result inlcude predict label, embedding map(random projection), visual attention map
            ['c' key]: change mode, 'auto predict' or 'press 'p' to predict'
            ['b' key]: change to before image
            ['a' key]: change to after image
            ['r' key]: reset current state
"""