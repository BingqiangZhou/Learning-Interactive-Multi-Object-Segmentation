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