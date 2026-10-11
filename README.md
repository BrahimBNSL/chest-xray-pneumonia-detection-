# Détection de pneumonie sur radiographies thoraciques

Projet de deep learning qui classe des radiographies thoraciques en **normal** ou **pneumonie**, avec un ResNet-50 pré-entraîné, puis une application web qui permet d'analyser une image envoyée par l'utilisateur.

> **Avertissement.** Ce projet est une démonstration académique et un outil d'aide au tri. Il n'est pas un dispositif médical et ne remplace ni l'avis d'un radiologue ni un examen clinique. Il n'a été ni validé en clinique, ni évalué sur d'autres populations que celle de son jeu de données.

## Objectif

Évaluer ce qu'un réseau pré-entraîné peut apprendre sur des radiographies, et mesurer honnêtement sa performance : comparaison à une référence sans adaptation, stabilité sur plusieurs entraînements, et limites du jeu de données.

## Données

Le projet utilise le jeu **Chest X-Ray Pneumonia** (Kermany et al., 2018), disponible sur Kaggle. Il contient environ 5 800 radiographies de patients pédiatriques (1 à 5 ans), réparties en deux classes.

Les données ne sont pas incluses dans ce dépôt. Elles doivent être téléchargées séparément et placées dans `data/chest_xray/`.

Points à connaître :
- Le déséquilibre est fort dans le train (74 % de pneumonies) et plus faible dans le test (62,5 %).
- Les noms de fichiers ne donnent qu'un identifiant patient approximatif. Des chevauchements patients entre train et test sont donc possibles.

## Méthode

1. **Exploration** (`notebooks/01_exploration.ipynb`) : distribution des classes, intégrité des fichiers, formats, luminosité.
2. **Données** (`src/data.py`) : conversion en niveaux de gris sur 3 canaux, redimensionnement à 224 × 224, augmentation légère sur le train seulement, découpage de la validation par groupe avec stratification, poids de classe calculés sur le train.
3. **Référence** (`src/features.py`) : ResNet-50 entièrement gelé, puis régression logistique sur ses caractéristiques.
4. **Fine-tuning** (`src/train.py`) : ResNet-50 entièrement réentraîné, avec AdamW à deux taux d'apprentissage, planification cosinus, perte pondérée, précision mixte, et arrêt anticipé sur la perte de validation.
5. **Stabilité** (`src/ensemble.py`) : trois entraînements avec des graines différentes, puis moyenne des probabilités.
6. **Application** (`app/`) : interface Gradio qui charge le modèle et analyse une image envoyée.

Le jeu de test n'a servi à aucune décision : ni le choix de C, ni le seuil, ni l'arrêt de l'entraînement.

## Résultats

Test de 624 images, seuil de décision 0,5.

**Référence sans adaptation** (ResNet-50 gelé + régression logistique)

| Métrique | Valeur |
|---|---|
| Balanced accuracy | 0,772 |
| ROC AUC | 0,950 |

**Fine-tuning complet** (graine 42, modèle de l'application)

| Métrique | Valeur |
|---|---|
| Balanced accuracy | 0,856 |
| ROC AUC | 0,961 |
| Accuracy | 0,888 |
| Rappel pneumonie | 0,982 (7 malades manqués sur 390) |
| Rappel normal | 0,731 (63 fausses alertes sur 234) |
| Précision pneumonie | 0,859 |

**Stabilité sur trois graines** (42, 1, 2)

| Métrique | Valeur |
|---|---|
| Balanced accuracy | 0,863 ± 0,013 |
| ROC AUC | 0,965 ± 0,004 |

**Ensemble des trois modèles** : balanced accuracy de 0,865, soit un gain non significatif par rapport à la moyenne. L'ensemble confirme surtout que la performance est reproductible.

### Matrice de confusion (graine 42)

| | Prédit normal | Prédit pneumonie |
|---|---|---|
| **Réel normal** (234) | 171 | 63 |
| **Réel pneumonie** (390) | 7 | 383 |

## Limites

- **Un seul jeu de données**, issu d'une même source, avec une population d'enfants de 1 à 5 ans. La performance sur des adultes, d'autres hôpitaux ou d'autres appareils est inconnue.
- **Fausses alertes élevées** : environ un cas sain sur quatre est classé malade. Le modèle est calibré pour privilégier le rappel de pneumonie.
- **Validation trop optimiste** : la balanced accuracy de validation (environ 0,97) dépasse nettement celle du test (0,86). La validation n'est donc pas une estimation fiable de la performance réelle.
- **Chevauchements patients possibles** entre train et test, faute d'identifiant patient fiable.
- **Pas d'explicabilité** pour l'instant : on ne vérifie pas encore si le modèle regarde les poumons, plutôt qu'un tube, une bordure ou le cadrage.

## Structure du projet

```
chest-xray-pneumonia/
├── app/
│   ├── app.py              # interface Gradio
│   └── predictor.py        # chargement du modèle et inférence
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── config.py           # chemins, hyperparamètres
│   ├── data.py             # chargement, transformations, découpage
│   ├── features.py         # référence : régression logistique
│   ├── model.py            # construction de ResNet-50
│   ├── train.py            # fine-tuning
│   ├── ensemble.py         # moyenne de plusieurs graines
│   └── utils.py            # graine, device, journalisation
├── outputs/                # checkpoints, métriques (non versionnés)
└── README.md
```

## Pistes d'amélioration

- Grad-CAM, pour vérifier que le modèle s'appuie sur les poumons.
- Test sur un jeu externe, comme RSNA Pneumonia Detection, adulte et au format DICOM.
- Second backbone, par exemple DenseNet-121, pour combiner des erreurs différentes.
- Validation croisée par patient, pour choisir le seuil sans regarder le test.

## Références

- Kermany, D. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5).
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Jeu de données : [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) sur Kaggle.