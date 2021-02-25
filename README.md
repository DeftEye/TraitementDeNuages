PFE : Analyse de l'organisation des nuages à partir d'images satellites
=========
Nous organisons notre avancé grace à un tableau  <a href="https://trello.com/b/S9iGb9xE/nuages-express"> Trello </a>.

Vous pouvez également retrouver plus d'informations sur notre projet dans le lien  <a href="https://www.kaggle.com/c/understanding_cloud_organization/overview"> Kaggle </a>.

Table des matières
============

<!--ts-->
   * [Contexte](#contexte)
   * [Approche](#approche)
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
