# PFE : Analyse de l'organisation des nuages à partir d'images satellites

```sh
https://trello.com/b/S9iGb9xE/nuages-express
```
## I.Contexte

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

# II.	Approche
- Prise en main la base de données d’images satellitaires,
- Etude sur les attributs pertinents à extraire des images,
- Implémentation des algorithmes de traitement d’images pour l’extraction d’attributs,
- Implémentation des réseaux convolutifs avec OpenCV et Pytorch pour la classification,
- Simulation et comparaison des résultats obtenus avec les différents modèles.


# III) References
- [1] Climatology of stratocumulus cloud morphologies: microphysical properties and radiative effects, Atmos. Chem. Phys - 2014
- [2] Combining crowd-sourcing and deep learning to explore themeso-scale organization of shallow convection , Rasp, Schulz, Bony and Stevens – 2019
- [3] Focal Loss for Dense Object Detection, Tsung-Yi Lin Priya Goyal Ross Girshick Kaiming He Piotr Dollar, Facebook AI Research (FAIR) - 2018


[Pytorch link] (https://pytorch.org/tutorials/)
[OpenCV link] (https://missinglink.ai/guides/computer-vision/opencv-deep-learning/)

> Encadrants CentraleSupélec : Jean-Luc Collette et Michel Ianotto
