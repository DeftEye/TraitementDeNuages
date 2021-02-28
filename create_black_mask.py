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
    width = 525;
    img = cv2.imread(inpath, cv2.IMREAD_COLOR)
    for i in range(0, height):
        for j in range(0, width):
            img[i][j] = (0, 0, 0)
    return img


for input_dir in ['./understanding_cloud_organization/train_images_350']:
    output_dir = input_dir + '_Black'

    os.makedirs(output_dir, exist_ok=True)

    filenames = list(os.listdir(input_dir))
    for filename in tqdm.tqdm(filenames):
        inpath = os.path.join(input_dir, filename)
        outpath = os.path.join(output_dir, filename)
        image = cv2.imread(inpath)
        image = create_masked_dataset(350, inpath, df_true)
        cv2.imwrite(outpath, image)