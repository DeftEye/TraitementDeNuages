# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import argparse
#
# import tqdm
# import pandas as pd
# import numpy as np

#
#
# def rle2mask(height, width, encoded):
#     img = np.zeros(height * width, dtype=np.uint8)
#
#     if isinstance(encoded, float):
#         return img.reshape((width, height)).T
#
#     s = encoded.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape((width, height)).T
import numpy as np
import cv2
from PIL import Image
from IPython.display import display


image = cv2.imread('./understanding_cloud_organization/train_images/bde641b.jpg', cv2.IMREAD_COLOR)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

for i in range(1, 100):
    for j in range(1, 100):
        image[i][j] = (0, 0, 255)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# MyImg = Image.new( 'RGB', (250,250), "black")
# pixels = MyImg.load() # creates the pixel map
# display(MyImg)        # displays the black image
# for i in range(MyImg.size[0]):
#     for j in range(MyImg.size[1]):
#         pixels[i,j] = (i, j, 100)
# MyImg - np.array(MyImg)
# cv2.imshow('image', MyImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()