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
df_mask_type = df['Label'] == 'Flower'
df_true = df[df_mask_type]

def create_masked_dataset(height, inpath, df):
    img = cv2.imread(inpath, cv2.IMREAD_COLOR)
    image_exact_name = inpath.split('/')[-1]
    df_mask_image = df['Image'] == image_exact_name
    df_real = df[df_mask_image]
    encoded = 0.2
    if df_real['EncodedPixels'].iloc[0] != np.NaN:
        encoded = df_real['EncodedPixels'].iloc[0]

    if isinstance(encoded, float):
        return img

    s = encoded.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    for lo, hi in zip(starts, ends):
        i, j, k = lo // height, lo % height, hi % height
        for l in range(j-1, k-1):
            img[l][i] = (20, 117, 255)
    return img


for input_dir in ['./understanding_cloud_organization/train_images_350_mask_Fish_Gravel_Sugar']:
    output_dir = input_dir + '_Flower'

    os.makedirs(output_dir, exist_ok=True)

    filenames = list(os.listdir(input_dir))
    for filename in tqdm.tqdm(filenames):
        inpath = os.path.join(input_dir, filename)
        outpath = os.path.join(output_dir, filename)
        image = cv2.imread(inpath)
        image = create_masked_dataset(350, inpath, df_true)
        cv2.imwrite(outpath, image)