# CV2 — Classification des graphiques : résultats réels

CV2 débloqué par les annotations officielles ; ResNet18 entraîné et évalué sur CPU.
Les anciens fichiers d'audit « sans labels » et le checkpoint technique sont
conservés pour l'historique. Le présent rapport et `results/cv2/final_report.json`
décrivent l'état final. CV3 et NLP non commencés.

## Source et contrôle des annotations

Source : https://huggingface.co/datasets/ahmed-masry/ChartQA
Révision figée : `af8b6f5c08c95085271561c2a3f9d15f2b5a9031`.
Fichier officiel : `ChartQA Dataset.zip`, champ JSON `type` des fichiers
`<split>/annotations/<image>.json`. Il ne s'agit pas du champ `label` des QA.
Archive distante : https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/af8b6f5c08c95085271561c2a3f9d15f2b5a9031/ChartQA%20Dataset.zip

Extraction HTTP Range : 22,996,493 octets transférés
(index ZIP et annotations), contre 875,370,872 octets pour
l'archive complète. Aucune image téléchargée. CRC ZIP et JSON validés.
Les 20,882 annotations sont stockées séparément dans
`annotations_cv2/af8b6f5c08c95085271561c2a3f9d15f2b5a9031/` avec `provenance.json`.
Toutes correspondent à une image locale décodable et à une des quatre classes.

63 références doublonnées écartées des listes de travail,
sans supprimer aucun fichier. Contrôle des pixels RGB décodés par SHA-256 et des
noms de fichiers ; priorité test puis validation puis train pour les doublons.
Parmi ces exclusions, 16 concernent des doublons entre splits.
Les splits d'origine sont conservés, aucune image n'est déplacée ou réaffectée.

| Split | v_bar | h_bar | line | pie | Total |
|---|---:|---:|---:|---:|---:|
| train | 10156 | 5391 | 2184 | 525 | 18256 |
| val | 495 | 340 | 171 | 48 | 1054 |
| test | 759 | 461 | 211 | 78 | 1509 |

Total utilisable : **20,819 images**.
`official_annotation_quality.json` contient distributions avant/après filtrage et
chaque exclusion ; `official_labels.csv` contient la liste utilisée ;
`official_image_hashes.json` conserve les empreintes.
Le lot de 40 images et `manual_labels.csv` restent intacts, inutilisés pour cet
entraînement officiel. Aucune prédiction employée comme vérité terrain.

## Prétraitement et modèle

`.venv` existante : Python 3.13.5, torch 2.12.0+cpu, torchvision 0.27.0+cpu,
scikit-learn 1.7.0. Aucun paquet réinstallé. CUDA indisponible ; 8 processeurs
logiques et environ 15 Go de RAM disponibles lors du contrôle, PyTorch limité
à 4 threads. Images RGB en 224×224, moyennes ImageNet (0.485, 0.456, 0.406),
écarts types (0.229, 0.224, 0.225), sans crop et sans binarisation CV1.

ResNet18 ImageNet (`IMAGENET1K_V1`) réutilise les poids déjà en cache.
Backbone et BatchNorm gelés, seule la couche finale à quatre sorties est entraînée.
Les caractéristiques du backbone sont calculées une fois puis mises en cache :
équivalent au transfert avec backbone gelé, sans augmentation aléatoire.
Les caches sont identifiés par les lignes du manifeste, empreintes d'images,
prétraitement et backbone. Adam, learning rate 0.001, CrossEntropyLoss non pondérée,
batch 32, seed 42, DataLoader Windows avec `num_workers=0`.

Documentation de l'architecture : https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18

## Essai et entraînement complet

Essai réel : 32 images train et 32 validation (8 par classe), 2
époques ; checkpoint rechargé et forward de validation réussi, rapport
`pilot_reload_test.json`. Accuracy validation finale : 59.375 %, loss 1.1114.
Cette mesure appartient au petit essai, pas au test final.

Entraînement complet : **20 époques**, 18,256
images train, 1,054 validation.
Temps enregistré après validation des fichiers, extraction des caractéristiques
incluse : 678.5 secondes.
Meilleure époque choisie sur la loss validation : **20**,
loss 0.000915, accuracy validation 99.9051%.
Le test n'a servi ni à l'optimisation ni au choix du checkpoint.

Modèle entraîné : `models_cv2/resnet18_20261117_010609_069660.pt`.
Il est distinct de `models_cv2/smoke_untrained.pt`, conservé et toujours refusé
pour une utilisation métier. Le checkpoint final inclut état du modèle, classes,
époque retenue, manifeste et empreintes. L'évaluation le recharge réellement et
recontrôle les images avant d'effectuer les prédictions.
Le contrôle `final_reload_test.json` confirme aussi des logits identiques entre
le modèle rechargé sur une image de validation et la tête appliquée à son cache.

## Évaluation indépendante

Test officiel filtré : **1509 images**, jamais utilisées pour entraîner.
Accuracy : **100.0000%**.

| Classe | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| v_bar | 1.0000 | 1.0000 | 1.0000 | 759 |
| h_bar | 1.0000 | 1.0000 | 1.0000 | 461 |
| line | 1.0000 | 1.0000 | 1.0000 | 211 |
| pie | 1.0000 | 1.0000 | 1.0000 | 78 |

Macro : précision 1.0000,
rappel 1.0000,
F1 **1.0000**.
Les moyennes pondérées sont également dans `metrics.json`.

Courbes train/validation : `results/cv2/train_20261117_010609_069660/loss.png`, `accuracy.png`, `history.json`,
`run_summary.json`. Les scores train sont des mesures pendant les mises à jour,
les scores validation sont calculés sans gradient après chaque époque.
Test : `results/cv2/test_20261117_010942_094601/metrics.json`, `confusion_matrix.png`, `predictions.json`,
`correct.png` et `incorrect.png` lorsqu'il existe des exemples correspondants.
Ce test n'a aucune erreur : aucun `incorrect.png` n'a été inventé.
`examples_by_class.png` illustre les quatre classes, dont les sorties ont été
revérifiées par des forwards directs. L'unique erreur de validation du meilleur
modèle (`multi_col_60411.png`, v_bar prédit h_bar, barres positives/négatives)
figure séparément dans `validation_incorrect.png` et `prediction_checks.json`.
Les métriques d'entraînement ne sont jamais présentées comme des métriques test.

## Commandes exécutées et relance dans VS Code

Depuis la racine du projet, terminal intégré PowerShell, utiliser l'exécutable
explicite de la `.venv` ; l'activation n'est pas nécessaire.
Ces scripts ont réellement été lancés depuis le terminal de l'agent :

```powershell
.\.venv\Scripts\python.exe -B cv/fetch_official_annotations.py --inspect
.\.venv\Scripts\python.exe -B cv/fetch_official_annotations.py
.\.venv\Scripts\python.exe -B cv/prepare_official_cv2.py
.\.venv\Scripts\python.exe -B -u cv/train_classifier.py --labels results/cv2/official_labels.csv --epochs 2 --batch-size 32 --max-per-class 8 --cache-features
.\.venv\Scripts\python.exe -B -u cv/train_classifier.py --labels results/cv2/official_labels.csv --epochs 20 --batch-size 32 --cache-features
.\.venv\Scripts\python.exe -B cv/evaluate_classifier.py --checkpoint "models_cv2/resnet18_20261117_010609_069660.pt"
.\.venv\Scripts\python.exe -B cv/inspect_cv2_results.py
.\.venv\Scripts\python.exe -B cv/report_cv2.py
```

Les téléchargements ont nécessité un accès réseau hors sandbox, sans changer les
réglages de sécurité. Aucun dataset supplémentaire ni poids ImageNet téléchargé
pendant cette finalisation. Le téléchargement reprend la révision courante de
Hugging Face ; le rapport présent identifie précisément celle utilisée ici.
Les fichiers d'annotation déjà présents ne sont pas écrasés.

L'inférence CLI a également été exécutée sur trois images test nouvelles pour
l'entraînement : `00339007006077.png` → h_bar, `00795994017065.png` → line,
`01001540004402.png` → h_bar. Commande réellement exécutée :

```powershell
.\.venv\Scripts\python.exe -B cv/chart_classifier.py --checkpoint "models_cv2/resnet18_20261117_010609_069660.pt" --image "ChartQA Dataset/test/png/00339007006077.png" "ChartQA Dataset/test/png/00795994017065.png" "ChartQA Dataset/test/png/01001540004402.png"
```

Pour prédire avec le modèle final (un ou plusieurs chemins après `--image`) :

```powershell
.\.venv\Scripts\python.exe -B cv/chart_classifier.py --checkpoint "models_cv2/resnet18_20261117_010609_069660.pt" --image "ChartQA Dataset/test/png/00339007006077.png"
```

Les nouveaux entraînements/évaluations créent des chemins horodatés. Le cache
réduit le coût des relances. Les scripts de préparation reconstruisent seulement
les listes et rapports CV2 ; ils n'écrivent pas dans les images originales.

## Fichiers créés ou modifiés pendant la finalisation

Créés : `cv/fetch_official_annotations.py`, `cv/prepare_official_cv2.py`,
`cv/inspect_cv2_results.py`, `cv/report_cv2.py`, `annotations_cv2/`, les listes et rapports officiels,
`results/cv2/pilot_reload_test.json`, `final_reload_test.json`, `final_report.json`, caches de caractéristiques,
deux nouveaux checkpoints entraînés, dossiers de courbes et d'évaluation.
Modifiés uniquement dans CV2 : `chart_classifier.py` (inférence multiple),
`prepare_cv2.py` (validation parallèle et empreintes), `train_classifier.py`
(petit essai, cache et traçabilité), `evaluate_classifier.py` (vérification du
checkpoint/des images et export des prédictions), ce README.
`cv/test_cv2_pipeline.py` reste disponible pour l'ancien test technique.

## Limites

Les annotations officielles peuvent être bruitées et n'ont pas toutes été
revérifiées humainement ; pas de vérité terrain inventée. Les doublons exacts et
les noms identiques sont exclus, mais les quasi-doublons sous noms et pixels
différents ne sont pas détectés : une indépendance sémantique parfaite n'est pas
garantie. Le corpus est déséquilibré et contient peu de secteurs ; consulter les
scores par classe et le F1 macro. Le resize peut déformer les proportions.
Il s'agit d'un transfert avec backbone gelé, pas d'un fine-tuning complet ; les
probabilités ne sont pas calibrées. Les scores ne couvrent que les quatre classes
ChartQA et ne garantissent pas la généralisation aux images extérieures.
Le test a été évalué après sélection sur validation ; aucune recherche
d'hyperparamètres basée sur le test n'a été réalisée.

Aucun fichier original ChartQA, notebook, fichier CV1 ou modèle initial modifié,
déplacé ou supprimé. Aucun push GitHub. Arrêt après CV2.
