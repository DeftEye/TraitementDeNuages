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

import PIL.Image as Image
import cv2
from IPython.display import display
from sklearn.model_selection import train_test_split
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

classes = {
        'Fish': 0,
        'Flower': 1,
        'Gravel': 2,
        'Sugar': 3
        }

#Dataset train

csv_train_path = './train_350.csv'
df_train = pd.read_csv(csv_train_path)
#df_train = df_train[df_train['Image_Label'].str.contains('jpg')==True]
#print(df_train.head(77))

#Retourne df avec Classes, labels, Largeur(list), Hauteur(list), taille(liste) ,(ajouter centre x,y)
def def_new_df():
    #Ajoute pour le label en mettant 1 ou 0 selon s'il y a ce label
    encodedpix_list = df_train['EncodedPixels'].values.tolist() # transforme en listes la colonne encodedpixels
    #Plus besoin de la columne exploitée
    df_train.drop(columns=['Image_Label'],inplace=True)
      
    #Retourne une liste avec des longueur, largeur par bounding box de classe
    count=0
    temp=""
    longueur_list=[]
    largeur_list =[]
    taille_list = []
    df_train["Longueur"] = np.nan
    df_train["Largeur"] = np.nan
    df_train["Taille"] = np.nan
    df_train["Emplacement_largest"]= np.nan
    df_train["cx"] = np.nan
    df_train["cy"]= np.nan
    #pour les bouding box rectangle sur cv2 il faut debut (x1,y1) et fin (x2,y2)
    df_train["x1"]= np.nan
    df_train["y1"]= np.nan
    df_train["x2"]= np.nan
    df_train["y2"]= np.nan


    for i in range(len(encodedpix_list)):
        if(encodedpix_list[i] is not np.nan): # seulement pour les valeurs non nulles
            encodedpix_list[i] =encodedpix_list[i].split(' ') # séparer le string en une liste encodepix_list     
            #pour chaque element de la liste de la colonne i
            #Methode 1
            for j in range(1, len(encodedpix_list[i]), 2 ):
                    
                if(j==1):
                    templong = encodedpix_list[i][1] #valeur long  
                    templarg=0
                    tempmult=0 
                    tempempl=encodedpix_list[i][0]
                    cx= 0
                    cy= 0
                    templongmax=0 
                    tempemplmax=0

                if(encodedpix_list[i][j]==templong):
                    count=count+1
  
                if (encodedpix_list[i][j]!= templong):
                    multiplication = int(encodedpix_list[i][j-2])*count  

                    if (multiplication>tempmult):
                        tempmult=multiplication
                        templarg=count
                        templongmax=encodedpix_list[i][j-2] 
                        tempemplmax=tempempl
                        #taille image de base : 525*325
                        cy= math.ceil(int(tempempl)%325+ int(templarg)/2)
                        cx= math.ceil(math.ceil(int(tempempl)/325) + int(templong)/2)

                    count=0
                    templong = encodedpix_list[i][j]
                    tempempl= encodedpix_list[i][j-1]

                if ((j == len(encodedpix_list[i])-1) and (templarg==0)):
                     templarg = count
                     templongmax= templong
                     tempemplmax = tempempl
                     tempmult = int(templong)*count
                     cy= math.ceil(int(tempemplmax)%325+ int(templarg)/2)
                     cx= math.ceil(math.ceil(int(tempemplmax)/325) + int(templong)/2)     

                if (j == len(encodedpix_list[i])-1):
                    df_train.iloc[i, 5] = templongmax
                    df_train.iloc[i, 6] = templarg
                    df_train.iloc[i, 7] = tempmult
                    df_train.iloc[i, 8] = tempemplmax
                    df_train.iloc[i, 9] = cx
                    df_train.iloc[i, 10] = cy

                    df_train.iloc[i, 11] = int(cx)-int(0.5*int(templongmax))
                    df_train.iloc[i, 12] = int(cy)+int(0.5*int(templarg))
                    df_train.iloc[i, 13] = int(cx)+int(0.5*int(templongmax))
                    df_train.iloc[i, 14] = int(cy)-int(0.5*int(templarg))
                    count=0
    df_train.drop(columns=['EncodedPixels'],inplace=True)

def filter_largest(): # pour linstant juste par label 
    #list_id_image= key_images()
    df_train['Taille'].fillna(value=0, inplace=True)
    #df_train['Longueur'].fillna(value=0, inplace=True)
    #df_train['Largeur'].fillna(value=0, inplace=True)
    for i in range(0,df_train.shape[0],4) :
        maxi_idx=df_train.loc[i:i+3,'Taille'].idxmax(axis=1)
        largest_df.loc[i/4,:]=df_train.loc[maxi_idx,:]
        #df_train[i:i+4,:].where(df_train[i:i+4,'Taille']!=maxi).drop(axis=0,inplace= True)


###################MAIN#################
def_new_df()
#print(df_train.head(10))   

largest_df = pd.DataFrame(columns= ['Image','Label','LabelIndex','Fold','Longueur','Largeur','Taille','Emplacement_largest','cx','cy','x1','y1','x2','y2'])
filter_largest()
print(largest_df.head(8))  

#target_to_tensor("0011165") #Test sur l'image 0011165