PFE : Analyse de l'organisation des nuages à partir d'images satellites
=========
Nous organisons notre avancé grace à un tableau  <a href="https://trello.com/b/S9iGb9xE/nuages-express"> Trello </a>.

Vous pouvez également retrouver plus d'informations sur notre projet dans le lien  <a href="https://www.kaggle.com/c/understanding_cloud_organization/overview"> Kaggle </a>.

Table des matières
============

<!--ts-->
   * [Contexte](#contexte)
   * [Approche](#approche)
   * [Entrainement des modèles sur les clusters de CS Metz](#entrainement-des-modèles-sur-les-clusters-de-cs-metz)
   * [References](#references)
<!--te-->
Contexte
============
Les nuages peu profonds jouent un rôle important dans la détermination du climat de la terre mais il
est encore difficile de les intégrer dans les modèles climatiques. Les chercheurs en météorologie et
climatologie espèrent améliorer la compréhension physique de ces nuages en les classant en différents
types d’organisation.
Les nuages peuvent s'organiser de différentes façons et les frontières entre les différentes formes
d'organisation sont parfois floues. Cela rend difficile la construction d'algorithmes traditionnels basés
sur des règles pour séparer les différentes régions de nuages.
Le but de ce projet est de construire un classifieur basé sur les réseaux convolutifs permettant de
classer différentes formes d'organisation de nuages à partir d'images satellites. L’objectif final est de
mieux comprendre l’influence de ces nuages sur le climat et de développer des modèles de la
prochaine génération permettant de réduire les incertitudes liées aux projections climatiques.
La base de données fournie est composée d’images satellitaires comportant 4 types d’organisation de
nuages nommées : fleur, sable, gravier et poisson.

Approche
============
- Prise en main la base de données d’images satellitaires,
- Etude sur les attributs pertinents à extraire des images,
- Implémentation des algorithmes de traitement d’images pour l’extraction d’attributs,
- Implémentation des réseaux convolutifs avec OpenCV et Pytorch pour la classification,
- Simulation et comparaison des résultats obtenus avec les différents modèles.

Entrainement des modèles sur les clusters de CS Metz
============
Au préalable : 
- Avoir enregistrer sa clef ssh pour éviter d'avoir à se reconnecter
- Avoir téléchargerle script" <a href="https://raw.githubusercontent.com/jeremyfix/deeplearning-lectures/master/ClusterScripts/cscluster"> cluster </a> 
- Ici on aura le tutoriel pour l'utilisateur gpusdi1_34, il faut remplacer par l'user désiré

1) Réserver pour par exemple via uSkynet 100h
mama@mama-VirtualBox:~$ ./cscluster book -u cpusdi1_34 -c uSkynet -w 100:00
```
Reservation successfull 
 Booking requested : OAR_JOB_ID = 112807 
 Waiting for the reservation 112807 to be running, might last few seconds 
   The reservation 112807 is not yet running |  
```

2) Voir la réservation :
mama@mama-VirtualBox:~$ ./cscluster log -u gpusdi1_34 -c uSkynet
```Listing your current reservations 
Job id     Name           User           Submission Date     S Queue
---------- -------------- -------------- ------------------- - ----------
112807                    gpusdi1_34     2021-02-27 18:57:35 R default  
```

3) Se logger sur la réservation :
mama@mama-VirtualBox:~$ ./cscluster log -u gpusdi1_34 -c uSkynet -j 112807
```
Using OAR 
 Logging to the booked node 
 I am checking if the reservation 112807 is still valid 
    The reservation 112807 is still running 
Connect to OAR job 112807 via the node sh11
gpusdi1_34@sh11:~$ 
```

4) Utiliser byobu ( voir la <a href="https://doc.ubuntu-fr.org/"> documentation </a>. ): 
gpusdi1_34@sh11:~$ byobu
Quelques commandes importantes : 
- F3/F4 pour aller entre fenêtre d'une même session 
- Shift + flèche droitre/gauche pour naviguer entre split d'une fenêtre
- Ctrl+Shift+F2 pour une nouvelle session
- CTRL + F2 pour créer une nouvelle fenêtre verticale / Shift + F2 pour une fenêtre horizontale
- Alt + fleche du haut ou bas pour naviguer entre sessions
- CTRL + F6 pour supprimer une fenêtre

5) Entrainer le modele :
Commande selon action désirée

6) Quitter (en laissant l'entrainement continuer):
F6

7) Pour retourner sur l'entrainement : 
- Refaire à partir de l'étape 3 en se loggant sur la même réservation
- Relancher (étape 4) byobu et on aura notre session (à choisir)
![image](https://user-images.githubusercontent.com/55411197/109397027-db556880-7934-11eb-8457-ef46d6a03fd7.png)

Remarque : 
Finalement , après résultats, ne pas oublier de supprimer sa réservation
mama@mama-VirtualBox:~$ ./cscluster kill -u gpusdi1_34 -c uSkynet -j 112807


References
============
- [1] Climatology of stratocumulus cloud morphologies: microphysical properties and radiative effects, Atmos. Chem. Phys - 2014
- [2] Combining crowd-sourcing and deep learning to explore themeso-scale organization of shallow convection , Rasp, Schulz, Bony and Stevens – 2019
- [3] Focal Loss for Dense Object Detection, Tsung-Yi Lin Priya Goyal Ross Girshick Kaiming He Piotr Dollar, Facebook AI Research (FAIR) - 2018
- <a href="http://arxiv.org/abs/1611.10012"> [4] </a>  Huang, J., Rathod, V., Sun, C., Zhu, M., Korattikara, A., Fathi, A., … Murphy, K. (2016). Speed/accuracy trade-offs for modern convolutional object detectors. CoRR, abs/1611.10012
- <a href="https://medium.com/@jonathan_hui/what-do-we-learn-from-single-shot-object-detectors-ssd-yolo-fpn-focal-loss-3888677c5f4d"> [5] </a> Hui, J. (2018). What do we learn from region based object detectors (Faster R-CNN, R-FCN, FPN)
- <a href="http://arxiv.org/abs/1708.02002"> [6] </a> Lin, T., Goyal, P., Girshick, R. B., He, K., & Dollár, P. (2017). Focal loss for dense object detection
- <a href="http://arxiv.org/abs/1512.02325"> [7] </a> Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S. E., Fu, C., & Berg, A. C. (2015). SSD: single shot multibox detector. CoRR, abs/1512.02325
- <a href="https://github.com/DeftEye/TraitementDeNuages/blob/main/References/Fine_tuning.pdf"> [8] </a> Convolutional Neural Networks for Medical Image
Analysis: Full Training or Fine Tuning?, Nima Tajbakhsh, Member, IEEE, Jae Y. Shin, Suryakanth R. Gurudu, R. Todd Hurst, Christopher B. Kendall,
Michael B. Gotway, and Jianming Liang, Senior Member, IEEE, 2017

Tutoriels
- [Pytorch ](https://pytorch.org/tutorials/)
- [OpenCV](https://missinglink.ai/guides/computer-vision/opencv-deep-learning/)



> Encadrants CentraleSupélec : Jean-Luc Collette et Michel Ianotto
