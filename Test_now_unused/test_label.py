import os
import math
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import tqdm
import collections
from collections import Counter

classes = {
    'Fish': 0,
    'Flower': 1,
    'Gravel': 2,
    'Sugar': 3
}

# Dataset train

csv_train_path = '../understanding_cloud_organization/train.csv'
df_train = pd.read_csv(csv_train_path)


# df_train = df_train[df_train['Image_Label'].str.contains('jpg')==True]
# print(df_train.head(77))

# Retourne df avec Classes, labels, Largeur(list), Hauteur(list), taille(liste) ,(ajouter centre x,y)
def def_new_df():
    # Ajoute pour le label en mettant 1 ou 0 selon s'il y a ce label
    df_train['Image'] = df_train['Image_Label'].str.extract(r'(^[\s\S]{0,7})')  # 7 premiers caracteres = nom de limage
    df_train['Label'] = np.where(df_train['EncodedPixels'].notna(), 1, 0)  # presence ou nom du label
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


def key_images():  # "key" des id des images
    list_unique_id_image = df_train['Image'].unique()
    print(list_unique_id_image)
    return list_unique_id_image


# Fonction qui renvoie le largest_size toute classe confondue
# tel que (nouvelle colonne df_train['largest'] = valeur la plus grande en longueur*largeur
def filter_largest():  # pour linstant juste par label
    # list_id_image= key_images()
    df_train['Taille'].fillna(value=0, inplace=True)
    # df_train['Longueur'].fillna(value=0, inplace=True)
    # df_train['Largeur'].fillna(value=0, inplace=True)
    for i in range(0, df_train.shape[0], 4):
        maxi_idx = df_train.loc[i:i + 3, 'Taille'].idxmax(axis=1)
        largest_df.loc[i / 4, :] = df_train.loc[maxi_idx, :]
        # df_train[i:i+4,:].where(df_train[i:i+4,'Taille']!=maxi).drop(axis=0,inplace= True)


# Fonction qui transforme le target en tensor
"""
    Input :
        obj :{'bndbox': {}, 'class': 5}
    Output : two tensors,
                -  the first with [cx,cy,width, height]
                -  the second with [class]
    label = obj['class']
    bndbox = obj['bndbox']
    return {'bboxes': torch.Tensor([bndbox['cx'], bndbox['cy'], bndbox['width'], bndbox['height']]),
            'labels': torch.LongTensor([label])}
"""


def target_to_tensor(image_name):  # Pour l'image donnée (son nom unique)
    select_index_img = np.where(largest_df["Image"] == image_name)

    box_df = largest_df.loc[select_index_img[0], ['cx', 'cy', 'Longueur', 'Largeur']]
    temp = np.array(box_df.values, dtype=np.int32)
    box_tensor = torch.from_numpy(temp)

    label = largest_df.loc[select_index_img[0], ['Classes_index']]
    temp2 = np.array(label.values, dtype=np.int32)
    label_tensor = torch.from_numpy(temp2)

    print({'bboxes': box_tensor, 'label': label_tensor})

    # return {'bboxes':box_tensor,'labels':label_tensor}}


def make_image_transform(image_transform_params: dict, transform: object):
    """
    image_transform_params :
        {'image_mode'='shrink', output_image_size={'width':.., 'height': ..}}
    transform : a torchvision.transforms type of object
    """
    resize_image = image_transform_params['image_mode']

    if (resize_image == 'shrink'):
        preprocess_image = transforms.Resize((image_transform_params['output_image_size']['width'],
                                              image_transform_params['output_image_size']['height']))
    if preprocess_image is not None:
        if transform is not None:
            image_transform = transforms.Compose([preprocess_image, transform])
    return image_transform


def make_target_transform(target_transform_params: dict):
    """
        target_mode :
            largest_bbox  : outputs a tensor with the largest bbox (4 numbers)
    """
    target_mode = target_transform_params['target_mode']
    if target_mode == 'largest_bbox':
        t_transform = lambda target: target_to_tensor(
            filter_largest(target))  # Modify the get_bbox to keep only the largest bounding box
    return t_transform


def make_trainval_dataset(dataset_dir: str, image_transform_params: dict,
                          transform: object, target_transform_params: dict, download: bool):
    image_transform = make_image_transform(image_transform_params, transform)
    target_transform = make_target_transform(target_transform_params)

    dataset_train = VOC.VOCDetection(root=dataset_dir, image_set='train', transform=image_transform,
                                     target_transform=target_transform, download=download)
    dataset_val = VOC.VOCDetection(root=dataset_dir, image_set='val', transform=image_transform,
                                   target_transform=target_transform, download=download)
    return dataset_train, dataset_val


def_new_df()

print(df_train.head(10))

# key_images()

largest_df = pd.DataFrame(
    columns=['EncodedPixels', 'Image', 'Label', 'Classes', 'Classes_index', 'Longueur', 'Largeur', 'Taille',
             'Emplacement_largest', 'cx', 'cy'])
filter_largest()
print(largest_df.head(77))

target_to_tensor("0011165")  # Test sur l'image 0011165