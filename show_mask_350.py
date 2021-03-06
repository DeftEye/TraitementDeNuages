from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import cv2
import tqdm
import pandas as pd
import numpy as np


csv_train_path = './understanding_cloud_organization/train_350.csv'
df = pd.read_csv(csv_train_path)

number = 910

img_name = df.iloc[number][0]
img_path = './understanding_cloud_organization/train_images_350/' + img_name
encoded = df.iloc[number][2]
height = 350
width = 525

def rle2mask(height, width, encoded):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if isinstance(encoded, float):
        return img

    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    for lo, hi in zip(starts, ends):
        i, j, k = lo // height, lo % height, hi % height
        print(i,j,k)
        for l in range(j-1, k-1):
                img[l][i] = (0, 0, 255)
    return img

img = rle2mask(height, width, encoded)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
