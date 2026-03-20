# Projet AI detection in IoT
On teste des approches de machine learning basé sur le clustering et des arbres de décisions pour la détection d'attaques DDoS sur des objets IoTs

## Workflow
Commencez par générer l'environnement virtuelle avec 
```bash
uv sync
```

Puis activer l'environnement virtuel
```bash
source .venv/bin/activate
```

Enfin générer les deux datasets avec la commande
```bash
python main.py --make-datasets 
```


Vous avez ensuite le choix entre utiliser `main.py --metrics <metrics>` ou directement d'appeler un modèle de machine learning pour commencer l'entraînement sur un modèle spécifique pour la classification binaire ou multi-classes.

Pour appeler directement un modèle :
```bash
python main.py model <model_name>
```

ne rien préciser enclenchera l'entraînement pour la classification binaire et multi-classes pour spécifier l'un ou l'autre, il faut rajouter l'argument `-b | --binary` ou `-m | --multiclass`. Si vous souhaitez afficher la matrice de confusion, il faut ajouter l'argument `-c | --confusion-matrix`.
Pour sauvegarder en mémoire le modèle, ajouter l'argument `-s | --save`.

## Métriques
Les métriques utilisés et appelables pour l'affichage sont :
- Accuracy => argument : accuracy
- Precision => argument : precision
- Recall => argument : recall
- F1-score => argument : f1_score
- Taille en mémoire  => argument : file_size

## Modèles
On a 4 modèles de machine learning implémentés. 3 supervisés et 1 non supervisé.
- Random Forest
- Decision Tree
- KNN
- DBSCAN

### Options main.py
- --metrics < list des métriques> : ne rien spécifier les affichera tous
- --make-datasets : génère les datasets pour les deux types de classification
- -b | --binary : lance uniquement pour la classification binaire
- -m | --multiclass : lance uniquement pour la classification multiclass, mutuellement exclusif avec binaire. Il faut ne rien mettre pour lancer les deux
- model \<MODEL\> : lance seulement pour le modèle annoncé
  - RF DT KNN ou DBSCAN
  - -b | --binary : lance la classification binaire pour le modèle
  - -m | --multiclass : lance la classification multiclass pour le modèle. Pas mutuellement exclusif cette fois
  - -c | --confusion-matrix : montre la matrice de confusion
  - -s | --save : sauvegarde le modèle en mémoire