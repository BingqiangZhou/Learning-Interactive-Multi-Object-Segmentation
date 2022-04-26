from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import random_projection

# refernece: https://matplotlib.org/stable/tutorials/colors/colormaps.html
def set_rgb_color_map_by_name(N=9, color_map_name='Set1',zero_to_255=True):
    color_map = plt.get_cmap(color_map_name)
    if N > color_map.N:
        print(f"colormap '{color_map_name} only have {color_map.N} color'")
        N = color_map.N
    x = np.linspace(start=0.0, stop=1.0, num=N)
    rgb = cm.get_cmap(color_map)(x, bytes=zero_to_255)[:, :3] # RGBA (N, 4)
    return rgb

def random_project_to_n_channel(embedding_np, out_channel=3):
    """
        embedding: (c, h, w)
    """
    c, h, w = embedding_np.shape
    transformer = random_projection.GaussianRandomProjection(n_components=out_channel)
    
    embedding_temp = np.transpose(embedding_np, (1, 2, 0)) #  (c, h, w) -> (h, w, c)
    embedding_temp = embedding_temp.reshape(h * w, c) # (h, w, c) -> (h * w, c)
    embedding_temp = transformer.fit_transform(embedding_temp) # (h * w, c) -> (h * w, 3)
    embedding_temp = embedding_temp.reshape(h, w, out_channel)  # (h * w, 3) -> (h, w, 3)
    embedding = (embedding_temp - embedding_temp.min()) / (embedding_temp.max() - embedding_temp.min())

    return embedding # (h, w, 3)

def bitget(number, pos):
    return (number >> pos - 1) & 1

def bitor(number1,number2):
    return number1 | number2

def bitshift(number,shift_bit_count):
    if shift_bit_count < 0:
        number = number >> abs(shift_bit_count)
    else:
        number = number << shift_bit_count
    return number 

def color_map(N, rgb_or_bgr='rgb'):
    cmap = np.zeros((N,3),np.uint8)
    for i in range(N): 
        id = i + 1 # 当id=0时，对应的颜色为（0，0，0），这里不包括（0，0，0）
        r = g = b = 0
        for j in range(8):
            r = bitor(r, bitshift(bitget(id,1),7 - j))
            g = bitor(g, bitshift(bitget(id,2),7 - j))
            b = bitor(b, bitshift(bitget(id,3),7 - j))
            id = bitshift(id,-3)
        if rgb_or_bgr == 'rgb':
            cmap[i,:]=[r,g,b]
        else:
            cmap[i,:]=[b,g,r]
    return cmap