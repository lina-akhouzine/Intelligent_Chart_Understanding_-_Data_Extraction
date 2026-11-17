# Chart2Data Intelligence — CV4 : extraction de données structurées

## Bilan honnête

Baseline fonctionnelle pour rectangles de **barres verticales**, extension
**horizontale expérimentale**. Courbes et secteurs non implémentés, explicitement
refusés (aucun `line_extractor.py`/`pie_extractor.py` fictif).
**25 graphiques du lot principal** : 20 v_bar et 5 h_bar, tous réellement traités.
11 images ont au moins une valeur numériquement appariée,
91 cellules appariées sur 324 références.
Couverture **28.09%** : la faible erreur conditionnelle ne signifie
pas que le système reconstruit tous les tableaux.

Des diagnostics supplémentaires ont traité un h_bar en pourcentage, un v_bar à
valeurs négatives, un line et un pie : **29 images distinctes** au total, dont les
25 du lot chiffré. Les diagnostics ne sont pas ajoutés artificiellement aux scores.
Les cinq images pilotes étaient déjà présentes dans CV3 ; quinze nouvelles images
v_bar ont été tirées avec seed 74 dans le test officiel. Les cinq h_bar viennent
du lot CV3. Le manifeste conserve le rôle de chaque image.

Le contrôle de rejet des pourcentages a été affiné après inspection du premier
passage et de ses erreurs : **évaluation exploratoire de développement**, pas un
benchmark indépendant jamais consulté. Aucune vérité terrain numérique ne sert
à produire une prédiction ni à corriger un chiffre OCR.

## Informations inspectées et séparation prédiction/référence

- Images sous `ChartQA Dataset/<split>/png/` : pixels des rectangles et textes.
- Checkpoint CV2 local `models_cv2/resnet18_20261117_010609_069660.pt` : inférence
  ResNet18 uniquement, pas d'entraînement ni téléchargement.
- Exports OCR CV3 : texte, boîte, confiance. Les rôles assistés par annotation et
  tout champ de référence sont ignorés ; la classification est recalculée.
- Annotations officielles `annotations_cv2/af8b6f5c08c95085271561c2a3f9d15f2b5a9031/` :
  type, `general_figure_info`, `models` (boîtes et valeurs sous-jacentes). Elles
  ont été inspectées, mais ne sont **jamais chargées par les modules prédictifs**.
- Tables originales sous `<split>/tables/*.csv` : première colonne catégorielle,
  colonnes de séries, nombres parfois suivis de `%`. Lecture par l'évaluateur
  **seulement après sauvegarde de la prédiction**.
- `official_labels.csv` sert uniquement à choisir les types/splits du lot, pas
  à déterminer le type prédit ni une valeur.

`chart_data_extractor.py` et `bar_extractor.py` ne lisent aucun CSV/JSON
 d'annotation pour extraire des valeurs. Les seuls JSON relus sont des caches OCR
vérifiés par SHA-256 de l'image. Les sorties CV4 conservent la provenance du cache
et `reference_data_used_for_prediction: false`.

## Algorithme réellement implémenté

1. Chargement RGB, SHA-256, CV2 en inférence. Si les rectangles présentent une
   base commune beaucoup plus cohérente dans l'autre orientation, correction
   géométrique explicite ; `cv2_chart_type` et `orientation_correction` conservent
   la prédiction originale et la justification. Ni annotation ni CSV ne sont lus.
2. Réutilisation OCR CV3/pipeline/CV4 par hash de l'image ; sinon appel au module
   CV3, variante originale. Il importe le prétraitement CV1 existant.
3. Quantification RGB et composantes OpenCV de couleurs pleines, filtres de taille,
   densité rectangulaire et position. Les gris sombres peuvent aussi être détectés.
4. Base candidate = bord partagé par le plus de rectangles. C'est une hypothèse
   géométrique, pas une détection garantie de toute ligne d'axe.
5. Bandes d'axes et catégories relues avec **le même lecteur EasyOCR**, sans rotation
   forcée, sur les pixels uniquement. Ces lectures ciblées restent dans `refined_ocr`.
6. Calibration robuste linéaire depuis au moins trois valeurs distinctes : signe
   de pente attendu, couverture spatiale, consensus >=75 %, rejet d'hypothèses
   concurrentes et unités contradictoires.
7. Conversion directe `valeur = pente × coordonnée du bord + intercept`.
   Aucune hypothèse de base égale à zéro. Les coordonnées hors de la plage
   calibrée sont refusées au-delà d'une petite marge explicitée dans le code.
8. Catégories associées par proximité géométrique. Noms de séries associés uniquement
   si un marqueur de légende de la bonne couleur est trouvé près du texte OCR.
   Sinon `series_1`, etc. sont des **identifiants techniques**, pas des noms inventés.
9. Si plusieurs rectangles occupent une même cellule, sa valeur devient manquante,
   sans somme ni moyenne automatique. Une barre non détectée n'est pas un zéro.

### Cas difficiles

- **Origine non nulle** : formule affine vérifiée sur un graphique synthétique
  de base 100. Pas de validation exhaustive d'axes tronqués sur images réelles.
- **Valeurs négatives** : choix du bord opposé à la base, signe par calibration.
  Vérifié synthétiquement. Le cas réel `multi_col_60411.png` retrouve l'orientation
  v_bar mais échoue à calibrer les graduations OCR : aucune valeur produite.
- **Pourcentages** : `12.5%` reste 12.5, unité `percent` (points de pourcentage,
  pas fraction 0.125). Le h_bar `00339007006077.png` extrait réellement quatre
  valeurs : Libya ≈5.315, Morocco ≈5.100, Lebanon ≈4.497, Colombia ≈1.449.
- Le `%` est souvent lu `70`, `76`, `25`, etc. Deux garde-fous **refusent** les
  échelles contradictoires : pourcentages de valeurs visibles, puis corrélation
  avec des gabarits du glyphe `%` dans les graduations. Aucune division par 100
  ni correction des chiffres n'est devinée.
- Le contrôle visuel utilise Arial Windows (`C:/Windows/Fonts/arial.ttf`, tailles
  8–18, seuil conservateur 0.50 dans >=3 graduations et >=60 % des graduations).
  Il peut produire de faux rejets et ne se transfère pas automatiquement à toute
  police. S'il n'y a pas cette police, ce contrôle est indisponible : il ne faut
  pas supposer la même robustesse sur une autre machine.
- **Séries groupées** : réellement extraites sur `multi_col_438.png`, 20 cellules
  appariées, légendes `Grooms`/`Brides`. Petites séries et séries sans légende restent
  partielles ; une série non nommée n'est pas appariée arbitrairement à une référence.
- **Empilements** : deux cas refusés dans le lot ; aucune décomposition implémentée.
- **Logarithmes/non-linéarité/ambiguïté** : pas de conversion non linéaire. Rejet
  si les graduations ne supportent pas l'affine ; test synthétique logarithmique.
  Cela ne garantit pas l'identification de toute échelle non linéaire clairsemée.
- **Unités** : symboles monétaires/`%` explicites et multiplicateurs de graduations
  parsés. Une unité seulement mentionnée dans un titre non reconnu reste inconnue.
  Les unités financières/monétaires et multiplicateurs de titres ne sont pas
  normalisés universellement entre images ; vérifier la compatibilité avant usage.

## Fichiers et exports

Créés uniquement pour CV4 :

- `cv/chart_data_extractor.py` : orchestration image/CV2/OCR et exports.
- `cv/bar_extractor.py` : rectangles, calibration, catégories et légendes.
- `cv/evaluate_extraction.py` : références exclusivement pour comparaison.
- `cv/test_cv4_pipeline.py` : tests synthétiques et vérification des sorties réelles.
- `cv/README_CV4.md` : ce rapport.
- `results/cv4/` : sorties horodatées et synthèse, aucun modèle supplémentaire.

Par image : `extraction.json` contient image, type, probabilités, catégories,
series/series_id, valeurs ou `null`, unités, coordonnées, calibration, provenance,
erreurs et incertitudes ; `data.csv` contient le DataFrame long, `table.csv` le
DataFrame pivoté catégories × séries ; `detected_elements.png` affiche rectangles
verts si valeur extraite, orange sinon, graduations bleues et base violette.
Une cellule absente du CSV est une valeur manquante, jamais un zéro automatique.
`success` signifie que tous les rectangles retenus ont une valeur, **pas** que
 toutes les valeurs de référence sont couvertes.

Les anciennes sorties exploratoires restent présentes pour respecter la consigne
sans suppression. **Le premier lot `evaluation_20261117_020522_908279` est dépassé**
et contient des erreurs d'échelle qui ont motivé les rejets : ne pas l'utiliser
comme résultat final. Prédictions finales :
`results/cv4/evaluation_20261117_021431_590936/`.
Évaluation finale recalculée sur ces prédictions sans relancer l'OCR :
`results/cv4/evaluation_20261117_021742_633345/metrics.json`.
`results/cv4/final_report.json` et `per_image_metrics.csv` donnent la synthèse et
les erreurs par image dans les unités natives.

## Évaluation et métriques réelles

Appariement par catégorie et série : texte Unicode normalisé, espaces normalisés,
marqueur final `*` retiré, correspondance exacte ou similarité textuelle unique
>=0.92 (écart >0.04 avec le second choix). **Aucun appariement basé sur la valeur
numérique**. Une seule série prédite et une seule colonne de référence autorisent
un appariement non ambigu même sans nom de série. Les collisions sont exclues.

MAE/RMSE : uniquement les cellules appariées ; MAPE : uniquement les paires dont
la référence est non nulle. Manquants et valeurs supplémentaires sont comptés
séparément, pas assimilés à zéro. Les paires conservent catégorie, série,
prédiction, référence et erreur. Les cellules non numériques de référence sont
exclues du dénominateur numérique. Les hypothèses de compatibilité d'unités natives
restent à vérifier hors du lot inspecté.

| Mesure globale | Résultat |
|---|---:|
| Images | 25 |
| Valeurs appariées / références | 91 / 324 |
| Valeurs manquantes | 233 |
| Valeurs supplémentaires/non appariées | 5 |
| MAE conditionnelle | 11.913705 |
| RMSE conditionnelle | 35.727779 |
| MAPE, références non nulles | 0.5864 % (91 paires) |
| Correspondance des catégories | 84.93% |
| Couverture des valeurs | 28.09% |
| Images avec une valeur extraite | 12/25 (48%) |
| Images avec une valeur appariée | 11/25 (44%) |
| Couverture complète des cellules sans extra | 20% |

**MAE/RMSE globales mélangent des unités natives différentes** et ne doivent pas
être interprétées comme une erreur physique universelle. Les métriques par image
sont dans `per_image_metrics.csv` et `per_image[].evaluation.numerical_metrics`.
Le MAPE conditionnel n'est pas une mesure de performance des cellules manquantes.

| Type | Images | Cellules appariées/références | Couverture | MAPE conditionnel |
|---|---:|---:|---:|---:|
| h_bar | 5 | 6/48 | 12.50% | 0.404 % |
| v_bar | 20 | 85/276 | 30.80% | 0.599 % |

Treize images n'ont aucune valeur extraite : deux empilements, trois rejets du
contrôle visuel `%`, un conflit avec les pourcentages visibles, et sept autres
échecs de calibration (graduations insuffisantes/incohérentes/ambiguës).
Les 15 nouvelles images aléatoires restent un sous-ensemble exploratoire :
77 paires, couverture
33.05%, MAPE
0.5628 %, après inspection de développement.

### Exemples automatiquement reconstruits

`two_col_3752.png`, contenu réel de `table.csv` :

```csv
category,series_1
Apr '16,18.874666819289345
Jul '14*,11.841487743855595
```

Comparaison d'évaluation uniquement : références 19 et 12, jamais injectées dans
l'extracteur. `multi_col_438.png` reconstruit notamment 2010 : Grooms ≈29.805,
Brides ≈27.596, et 2019 : Grooms ≈30.225, Brides ≈28.648. Les visualisations ont
été ouvertes et inspectées : frontières des barres, graduations et base cohérentes.

## Exécution réelle et relance PowerShell dans VS Code

Aucune dépendance ajoutée : `.venv` CV3 réutilisée, Python 3.13.5, CPU,
PyTorch 2.12.0, OpenCV 4.13.0, EasyOCR 1.7.2, NumPy/Pandas/Pillow existants.
Les poids OCR et ResNet18 sont locaux, `download=False`.
Depuis la racine du projet :

```powershell
# Exemple vertical (commande pilote exécutée avec cette image et quatre autres)
.\.venv\Scripts\python.exe -B -u cv/chart_data_extractor.py --image "ChartQA Dataset/test/png/two_col_3752.png"
# Exemple horizontal réellement exécuté
.\.venv\Scripts\python.exe -B -u cv/chart_data_extractor.py --horizontal --image "ChartQA Dataset/test/png/00339007006077.png" "ChartQA Dataset/val/png/multi_col_60411.png"
# Lot de 20 verticales + 5 horizontales, réellement exécuté deux fois
.\.venv\Scripts\python.exe -B -u cv/evaluate_extraction.py --count 20 --horizontal
# Recalcul des métriques uniquement, commande finale exécutée
.\.venv\Scripts\python.exe -B -u cv/evaluate_extraction.py --reuse-run results/cv4/evaluation_20261117_021431_590936
# Tests mathématiques puis validation réelle, commandes exécutées
.\.venv\Scripts\python.exe -B cv/test_cv4_pipeline.py --synthetic-only
.\.venv\Scripts\python.exe -B cv/test_cv4_pipeline.py --evaluation results/cv4/evaluation_20261117_021742_633345
```

Le contrôle de refus a aussi été réellement exécuté :

```powershell
.\.venv\Scripts\python.exe -B -u cv/chart_data_extractor.py --image "ChartQA Dataset/test/png/10146.png" "ChartQA Dataset/test/png/two_col_4223.png"
```

Les deux sorties sont `unsupported`, sans valeurs. Les commandes de l'agent ont
été lancées dans son terminal ; les mêmes commandes sont utilisables dans le
terminal intégré VS Code. Aucun entraînement, installation, téléchargement ou
push. Le lanceur global des étapes précédentes n'a pas été modifié.

Utilisation Python avec les imports existants du projet :

```python
import sys
sys.path.insert(0, 'cv')
from chart_data_extractor import ChartDataExtractor, dataframes
result, rgb = ChartDataExtractor().extract('ChartQA Dataset/test/png/two_col_3752.png')
long_df, wide_df = dataframes(result)
```

## Tests et limites avant NLP

Rapport final des tests : `results/cv4/tests_20261117_021840_299199/test_report.json` — **PASS**.
Les 25 exports JSON, CSV et visualisations ont été relus ; comptes et valeurs
manquantes contrôlés ; hash des images inchangé ; aucune valeur si calibration
absente. Les empreintes des fichiers préexistants CV1–CV3 et du checkpoint sont
inchangées pendant les tests. Le test initial supposait `run_all.py` présent,
alors qu'il est absent de cet espace de travail ; corrigé pour protéger uniquement
les fichiers effectivement présents, sans créer/remplacer ce lanceur.

Tests **synthétiques**, distincts des métriques réelles : échelle linéaire, origine
100, valeurs négatives, pourcentages, deux séries/légendes colorées, orientation
horizontale, rejet de logarithme/empilement/image blanche, parseur numérique strict,
exclusion des paires non appariées et références nulles dans le MAPE. Un placement
de légende synthétique hors région supportée a été corrigé dans la fixture du test.

Priorités avant d'envisager NLP : améliorer la lecture numérique et des symboles,
valider les unités, augmenter la couverture des graduations/catégories, détecter
les minuscules barres et zéros explicitement, prendre en charge les empilements,
les noms de séries et les axes tronqués réels, puis constituer un jeu indépendant
non consulté pendant le développement. La couverture horizontale actuelle reste
faible ; courbes et secteurs nécessitent des modules dédiés, pas une promesse.
Le cache OCR historique n'est pas intégralement versionné par paramètres du moteur.

Aucun fichier original ChartQA, annotation, fichier CV1/CV2/CV3 ou modèle existant
modifié, déplacé ou supprimé. Aucun fichier supprimé. Arrêt après CV4 ; NLP et
Streamlit non commencés.
