# RetT segmentation rivieres - Application à l'algorithme de débit HiVDI

## Test de l'équation Low-Froude sur une segmentation

Ce test nécessite de positionner les variables d'environnement suivantes:
    - PEPSI1_DIR : chemin vers le dossier contenant les fichiers NetCDF du PEPSI challenge 1
    - PEPSI2_DIR : chemin vers le dossier contenant les fichiers NetCDF du PEPSI challenge 2

### Example de lancement
Lancement sur le cas GaronneUpstream avec les paramètres suivants:
    - segmentation baseline
    - lambda_c : 5 km
    - longueur minimale de reach : 1km
$ python lowfroude_estimations.py GaronneUpstream -segmentation baseline -lambda 5 -min-length 1

## Génération d'un cas test PEPSI avec nouvelle segmentation

Ce test nécessite de positionner les variables d'environnement suivantes:
    - PEPSI1_DIR : chemin vers le dossier contenant les fichiers NetCDF du PEPSI challenge 1
    - PEPSI2_DIR : chemin vers le dossier contenant les fichiers NetCDF du PEPSI challenge 2
    
Ce test nécessite HiVDI (appel à des fonctions de HiVDI)

### Example de lancement
Lancement sur le cas OhioSection2 avec les paramètres suivants:
    - segmentation avancee
    - lambda_c : 5 km
    - longueur minimale de reach : 1km
    - fichier de sortie : OhioSection2_segmented.nc
$ python generate_segmented_pepsi_case.py OhioSection2 -segmentation advanced -lambda 5 -min-length 1 -o OhioSection2_segmented.nc

