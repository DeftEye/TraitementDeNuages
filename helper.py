import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

def plot_img_array(img_array, ncol=3):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])

from functools import reduce
def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

import itertools
def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()

def masks_to_colorimg(masks, haha):
    colors = np.asarray([(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 110, 255)])
    mask = masks
    if haha == "xD":
        mask = mask.numpy()
    mask = np.moveaxis(mask, -1, 0)
    mask = np.moveaxis(mask, -1, 0)

    colorimg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32) * 255
    height, width, channels = mask.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[mask[y,x,:] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)