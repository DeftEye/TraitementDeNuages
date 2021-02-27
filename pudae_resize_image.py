from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tqdm
import cv2


def resize(image, size=(525, 350)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return image


for input_dir in ['./understanding_cloud_organization/test_images', './understanding_cloud_organization/train_images']:
    output_dir = input_dir + '_350'

    os.makedirs(output_dir, exist_ok=True)

    filenames = list(os.listdir(input_dir))
    for filename in tqdm.tqdm(filenames):
        inpath = os.path.join(input_dir, filename)
        outpath = os.path.join(output_dir, filename)
        image = cv2.imread(inpath)
        image = resize(image)
        cv2.imwrite(outpath, image)