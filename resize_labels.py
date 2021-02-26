from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import json
import urllib
import PIL.Image as Image
import cv2
import torch
import torchvision
from glob import glob
from IPython.display import display
from sklearn.model_selection import train_test_split

import seaborn as sns
from pylab import rcParams, matplotlib
import matplotlib.pyplot as plt
import understanding_cloud_organization
from understanding_cloud_organization import train_images
from matplotlib import rc


classes = {
    'Fish': 0,
    'Flower': 1,
    'Gravel': 2,
    'Sugar': 3
}

# Dataset train

csv_train_path = './understanding_cloud_organization/train.csv'
df_train = pd.read_csv(csv_train_path)

def def_new_df():
    # Ajoute pour le label en mettant 1 ou 0 selon s'il y a ce label
    df_train['Image'] = df_train['Image_Label'].str.extract(r'(^[\s\S]{0,7})')  # 7 premiers caracteres = nom de limage
    df_train['Presence'] = np.where(df_train['EncodedPixels'].notna(), 1, 0)  # presence ou nom du label
    df_train['Classes'] = df_train.Image_Label.map(lambda v: v[v.find('_') + 1:])  # classe de l'image
    df_train['Classes_index'] = [classes.get(v, None) for v in df_train['Classes']]  # index de la classe
    encodedpix_list = df_train['EncodedPixels'].values.tolist()  # transforme en listes la colonne encodedpixels

    # Plus besoin de la columne exploitée
    df_train.drop(columns=['Image_Label'], inplace=True)

    # Retourne une liste avec des longueur, largeur par bounding box de classe
    count = 0
    temp = ""
    longueur_list = []
    largeur_list = []
    taille_list = []
    df_train["Longueur"] = np.nan
    df_train["Largeur"] = np.nan
    df_train["Taille"] = np.nan
    df_train["Emplacement_largest"] = np.nan
    df_train["cx"] = np.nan
    df_train["cy"] = np.nan

    for i in range(len(encodedpix_list)):
        if (encodedpix_list[i] is not np.nan):  # seulement pour les valeurs non nulles
            encodedpix_list[i] = encodedpix_list[i].split(' ')  # séparer le string en une liste encodepix_list
            # pour chaque element de la liste de la colonne i
            # Methode 1
            for j in range(1, len(encodedpix_list[i]), 2):

                if (j == 1):
                    templong = encodedpix_list[i][1]  # valeur long
                    templarg = 0
                    tempmult = 0
                    tempempl = encodedpix_list[i][0]
                    cx = 0
                    cy = 0
                    templongmax = 0
                    tempemplmax = 0

                if (encodedpix_list[i][j] == templong):
                    count = count + 1

                if (encodedpix_list[i][j] != templong):
                    multiplication = int(encodedpix_list[i][j - 2]) * count

                    if (multiplication > tempmult):
                        tempmult = multiplication
                        templarg = count
                        templongmax = encodedpix_list[i][j - 2]
                        tempemplmax = tempempl
                        # taille image de base : 2100*1400
                        cy = math.ceil(int(tempempl) % 1400 + int(templongmax) / 2)
                        cx = math.ceil(math.ceil(int(tempempl) / 1400) + int(templarg) / 2)

                    count = 0
                    templong = encodedpix_list[i][j]
                    tempempl = encodedpix_list[i][j - 1]

                if ((j == len(encodedpix_list[i]) - 1) and (templarg == 0)):
                    templarg = count
                    templongmax = templong
                    tempemplmax = tempempl
                    tempmult = int(templong) * count
                    cy = math.ceil(int(tempemplmax) % 1400 + int(templong) / 2)
                    cx = math.ceil(math.ceil(int(tempemplmax) / 1400) + int(templarg) / 2)

                if (j == len(encodedpix_list[i]) - 1):
                    df_train.iloc[i, 5] = templongmax
                    df_train.iloc[i, 6] = templarg
                    df_train.iloc[i, 7] = tempmult
                    df_train.iloc[i, 8] = tempemplmax
                    df_train.iloc[i, 9] = cx
                    df_train.iloc[i, 10] = cy
                    count = 0






def_new_df()
df_train.pop('EncodedPixels')
df_train.pop('Emplacement_largest')

df_train.groupby('Image')
df_train.to_csv(path_or_buf= 'test1.csv', index=False)
# fichier = open("new_csvv2.csv", "a")
# for item in df_train.to_dict('records'):
#     fichier.write("%s\n" % item)
#     fichier.write(",")
# fichier.close()





# def load_image(img_path, resize=True):
#   img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#
#   if resize:
#     img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
#
#   return img
#
# def show_image(img_path):
#   img = load_image(img_path)
#   plt.imshow(img)
#   plt.axis('off')
#
# def show_sign_grid(image_paths):
#   images = [load_image(img) for img in image_paths]
#   images = torch.as_tensor(images)
#   images = images.permute(0, 3, 1, 2)
#   grid_img = torchvision.utils.make_grid(images, nrow=11)
#   plt.figure(figsize=(24, 12))
#   plt.imshow(grid_img.permute(1, 2, 0))
#   plt.axis('off')
#
# sample_images = np.random.choice(glob('./understanding_cloud_organization/train_images/*.jpg'), 5)
# print(sample_images)
# show_sign_grid(train_images)

