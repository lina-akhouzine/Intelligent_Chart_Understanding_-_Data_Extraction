# CV3 — Extraction OCR de ChartQA

CV3 lit les **textes visibles**, leurs boîtes et leurs confiances. Les chaînes
numériques reconnues ne sont pas des données extraites des barres/courbes : CV4
n'est pas commencé. Les scripts et modèles CV1/CV2 sont uniquement importés/lus.

## Environnement et moteur

Windows, Python 3.13.5, `.venv` existante. EasyOCR 1.7.2 fonctionne réellement
avec PyTorch 2.12.0+cpu et Torchvision 0.27.0+cpu. CUDA indisponible, quatre threads.
Choix EasyOCR : installation et initialisation réussies sur cette machine ; il
n'a pas été nécessaire d'essayer PaddleOCR ou une alternative externe.

Dépendances manquantes installées dans la `.venv` : scikit-image 0.26.0,
python-bidi 0.6.12, shapely 2.2.0, pyclipper 1.4.0, ninja 1.13.2, ainsi que les
dépendances imageio 2.38.2, lazy-loader 0.6 et tifffile 2026.11.12.
EasyOCR a été installé avec `--no-deps` après ces dépendances : son exigence
déclarative `opencv-python-headless` est satisfaite fonctionnellement par le
module `cv2` de l'OpenCV déjà présent. Aucune seconde distribution OpenCV n'a été
installée pour éviter le masquage du module utilisé par CV1/CV2. Un outil qui
vérifie seulement les noms des distributions peut donc signaler cette exigence
headless comme absente, malgré les imports et traitements réussis.

Poids OCR officiels téléchargés séparément :

- `models_cv3/easyocr/weights/craft_mlt_25k.pth` : 83 152 330 octets.
- `models_cv3/easyocr/weights/english_g2.pth` : 15 143 997 octets.

Le lecteur utilise `['en']`, adapté aux graphiques anglais du lot. Il ne promet
pas de reconnaître toutes les langues. Les téléchargements ne sont autorisés
que sur demande explicite du script (`--download-models`) ; l'évaluation réutilise
les poids locaux. L'avertissement PyTorch sur `pin_memory` sans GPU est bénin.
Documentation : https://www.jaided.ai/easyocr/documentation/
Source : https://github.com/JaidedAI/EasyOCR

## Annotations réellement disponibles

Source officielle déjà téléchargée :
https://huggingface.co/datasets/ahmed-masry/ChartQA
Révision `af8b6f5c08c95085271561c2a3f9d15f2b5a9031`.
Audit de **20 882 JSON** dans `annotations_cv2/` ; rapport exhaustif dans
`results/cv3/annotation_audit.json`.

| Champ sous `general_figure_info` | Fichiers où la clé existe | Contenu |
|---|---:|---|
| `title.text`, `title.bbox` | 2 310 | Titre et boîte ; 2 309 boîtes exploitables |
| `figure_info.bbox` | 19 865 | Région du tracé, pas une transcription |
| `x_axis.major_labels.values` et `.bboxes` | 19 591 | Catégories et boîtes |
| `y_axis.major_labels.bboxes` | 18 562 | Boîtes des graduations |
| `y_axis.major_labels.values` | 1 007 | Valeurs textuelles explicitement présentes |
| `y_axis.label.text`, `.bbox` | 17 555 | Titre de l'axe sémantique Y |
| `x_axis.label` | 17 555 | Dictionnaire vide dans cette version : pas de `.text` |
| `legend.items` | 4 624 | Objets avec `label.text` et `label.bbox` quand présents |
| `legend.bbox` | 3 315 | Enveloppe de la légende, pas chaque texte |

`legend.items[].preview.bbox` décrit le symbole, pas le texte. Certaines tranches
de secteurs ont `models[].text_label` et `models[].text_bbox` : ces deux champs
explicitement couplés sont utilisables comme texte visible, avec rôle `other`.
`models[].x`, `.y`, `.value`, `.name` ne sont jamais transformés en vérité terrain
OCR. Les annotations ne sont ni complétées ni corrigées automatiquement.

Le lecteur d'annotations obtient **395 884 régions**, dont **260 460 avec
transcription**, et 20 251 fichiers ayant au moins une région transcrite.
Une boîte sans texte peut servir à évaluer la localisation et le rôle, mais pas
la reconnaissance. Les tailles de listes incohérentes sont signalées ; aucune
association texte/boîte n'est devinée. Les boîtes invalides sont ignorées et les
boîtes hors image sont exclues de l'évaluation du lot.

Les noms `x_axis`/`y_axis` sont souvent sémantiques : sur `h_bar`, les catégories
`x_axis` sont physiquement à gauche (Y), les graduations `y_axis` en bas (X).
Le parseur échange ces rôles pour les barres horizontales, sans changer les JSON.
Les références normalisées peuvent elles-mêmes être imparfaites ; elles ne
constituent pas une nouvelle annotation humaine exhaustive.

## Pipeline

`ocr_extraction.py` appelle `preprocessing.preprocess` de CV1 sans l'éditer.
Original RGB, gris, CLAHE, débruitage et binaire restent disponibles ; les essais
principaux OCR utilisent **original + CLAHE**, à la résolution native.
EasyOCR utilise `mag_ratio=1.5`, `canvas_size=2560`, `rotation_info=[90,270]` et
`min_size=5`. Ses coordonnées sont rapportées à l'image d'origine, puis bornées.

Les hypothèses de même région sont fusionnées si IoU >= 0.75 ou si IoU >= 0.45
et similarité textuelle >= 0.55. La plus confiante est gardée, avec les variantes
et alternatives conservées. Deux textes identiques à des positions distinctes
restent deux détections. La fusion de mots/lignes entre variantes reste imparfaite.
Aucun seuil de confiance ne cache les lectures faibles ; elles sont signalées.

Le checkpoint ResNet18 entraîné de CV2 est chargé en inférence uniquement pour
prédire le type. Ses poids ne sont ni modifiés ni réentraînés. Si indisponible,
le JSON contient `chart_type: null` et un avertissement.

`chart_text_parser.py` estime les rôles géométriquement : titre, titres d'axes,
catégories X/Y, graduations X/Y, autres, ou `unknown`. L'ajout de `y_category` et
`x_tick` évite de ranger les axes d'un graphique horizontal dans les mauvais
rôles. Les décisions et leur force heuristique sont exportées, sans prétendre
que `role_confidence` soit une probabilité calibrée.

Une région de légende ne peut pas être reconnue sûrement par la géométrie simple
actuelle : le rôle géométrique reste souvent `unknown` ou ambigu. Si une annotation
est fournie, `annotation_assistance` indique séparément son rôle, y compris
`legend`, et l'IoU. **Cette assistance ne remplace pas le rôle prédit et n'est
pas présentée comme une performance automatique.** Le texte de référence n'est
jamais donné au moteur OCR ni utilisé pour choisir les correspondances.

## Fichiers et sorties

- `cv/ocr_extraction.py` : lecteur réutilisable, fusion et CLI d'export.
- `cv/chart_text_parser.py` : annotations prudentes, géométrie et rôles.
- `cv/evaluate_ocr.py` : audit, échantillonnage et métriques.
- `cv/test_cv3_pipeline.py` : tests réels et contrôles des sorties.
- `cv/README_CV3.md` : ce document.
- `models_cv3/easyocr/` : uniquement les nouveaux poids OCR.
- `results/cv3/annotation_audit.json` : champs, références et incohérences.
- `results/cv3/evaluation_*/sample_manifest.json` : sélection reproductible.
- `results/cv3/evaluation_*/images/<image>.json` : image/source/SHA-256,
  dimensions, type prédit, textes, polygones et boîtes x1/y1/x2/y2 en pixels,
  confiances, rôles, variantes, alternatives, erreurs et incertitudes.
- `results/cv3/evaluation_*/images/<image>_boxes.png` : original annoté ; numéros
  bleus = index dans la liste JSON, orange = `unknown`, rouge = rôle estimé.
- `results/cv3/evaluation_*/metrics.json` : agrégats, par classe, par image,
  correspondances et exemples de bonnes/mauvaises reconnaissances.
- `results/cv3/ocr_*/` : exports des appels CLI indépendants.
- `results/cv3/tests_*/` : image blanche, image invalide et rapport de tests.

Les dossiers de traitement sont horodatés pour préserver les sorties précédentes.
L'audit reconstruit seulement son propre rapport CV3. Aucun fichier dataset,
annotation officielle ou fichier CV1/CV2 n'est écrit.

## Protocole d'évaluation

Lot fixé avant OCR : **20 images test, 5 par classe**, tirage seed 43 dans le
manifeste officiel CV2 dédoublonné. Aucun entraînement OCR ni choix de paramètres
par recherche sur ces images. L'assistance de régions est exportée séparément.

Appariement un-à-un hongrois, IoU >= 0.30, sans critère de texte. Les références
multi-lignes et les OCR segmentés différemment peuvent être mal appariés.

- CER/WER : micro-moyenne sur les régions explicitement transcrites, Unicode
  NFKC, minuscules (`casefold`), espaces normalisés, ponctuation conservée.
- Mesure principale : référence manquée = chaîne vide, donc suppressions.
- Mesure complémentaire : CER/WER uniquement sur les références appariées.
- Rappel détection : régions appariées / régions annotées évaluables.
- Précision **conditionnelle aux régions annotées** : régions appariées /
  détections OCR ayant IoU >= 0.10 avec une région de référence. Le texte non
  annoté est exclu, pas automatiquement considéré comme faux positif.
- Rôles : exactitude géométrique sur les paires localisées ; les rôles assistés
  par annotation ne sont pas évalués comme des prédictions.

Ces mesures ne sont **pas** une précision OCR exhaustive de la page, ni une
mesure de qualité des valeurs numériques des barres/courbes. CER/WER peuvent
dépasser 100 % si les insertions dépassent la longueur des références.

## Commandes PowerShell, racine du projet

Installation réellement exécutée (ne pas réinstaller si déjà présent) :

```powershell
.\.venv\Scripts\python.exe -m pip install scikit-image==0.26.0 python-bidi==0.6.12 shapely==2.2.0 pyclipper==1.4.0 ninja==1.13.2
.\.venv\Scripts\python.exe -m pip install --no-deps easyocr==1.7.2
```

Les poids ont été initialisés avec `easyocr.Reader(['en'], gpu=False, ...)`,
chemins explicites sous `models_cv3/easyocr`, depuis le terminal de l'agent.
Pour les télécharger si absents et lancer une extraction dans VS Code :

```powershell
.\.venv\Scripts\python.exe -B cv/ocr_extraction.py --image "ChartQA Dataset/test/png/00339007006077.png" --download-models
```

Audit et lot principal réellement exécutés :

```powershell
.\.venv\Scripts\python.exe -B -u cv/evaluate_ocr.py --audit-only
.\.venv\Scripts\python.exe -B -u cv/evaluate_ocr.py --per-class 5
```

CLI indépendante exécutée pour vérifier aussi le point d'entrée :

```powershell
.\.venv\Scripts\python.exe -B -u cv/ocr_extraction.py --image "ChartQA Dataset/test/png/00339007006077.png" --variants original
```

Tests et contrôle des exports :

```powershell
.\.venv\Scripts\python.exe -B cv/test_cv3_pipeline.py
```

L'option `--evaluation "results/cv3/evaluation_<horodatage>"` cible un lot précis.
Sans argument, le dernier lot ayant un `metrics.json` est utilisé. `-B` évite
les caches Python dans les dossiers des étapes précédentes. Les commandes sont
à reprendre dans le terminal intégré VS Code ; les exécutions de l'agent ont eu
lieu dans son terminal, sans prétendre contrôler celui de VS Code.

## Limites et améliorations avant CV4

Baseline OCR anglais, échantillon de 20 images seulement, annotations partielles
et parfois imparfaites. Petits caractères, inclinaisons, textes verticaux,
ponctuation, chiffres proches des barres et légendes sont difficiles. Une forte
confiance OCR n'est pas une garantie de justesse. Les variantes peuvent produire
des doublons résiduels lorsque leur segmentation diffère.

Le parseur géométrique est prudent et incomplet, en particulier pour légendes,
titres d'axes et mises en page inhabituelles. Il faut améliorer la détection du
tracé, le regroupement des mots en lignes, les crops et rotations d'axes, puis
valider les rôles sur des références mieux couvertes. Une revue humaine reste
nécessaire avant de consommer les nombres OCR comme graduations. Cela prépare
CV4 sans commencer l'extraction des valeurs ni le NLP.

Les résultats finaux mesurés figurent ci-dessous.


## Résultats finaux réellement mesurés

Lot OCR original : `results/cv3/evaluation_20261117_012518_614675/`.
Lot final après correction logique des rôles des secteurs (sans axes) :
`results/cv3/evaluation_20261117_013720_001076/`.
La reconnaissance OCR n'a pas été répétée ni retouchée : mêmes textes et scores,
avec provenance `roles_recomputed_from` dans les JSON. Commande réellement exécutée :

```powershell
.\.venv\Scripts\python.exe -B -u cv/evaluate_ocr.py --reuse-ocr results/cv3/evaluation_20261117_012518_614675
.\.venv\Scripts\python.exe -B cv/test_cv3_pipeline.py --evaluation results/cv3/evaluation_20261117_013720_001076
```

**20 graphiques évalués, 5 par classe, plus 1 graphique distinct testé via la CLI.**
Deux passes OCR par graphique du lot principal : original et CLAHE.
540 textes après fusion ; 287 régions de référence évaluables,
225 appariées ; 188 régions avec texte de référence,
181 appariées. 1889 caractères et 340 mots de référence.
295 détections entrent dans le dénominateur de précision conditionnelle.
Temps OCR cumulé du lot (prétraitement/initialisation hors chronomètre) :
670.5 secondes.

| Mesure | Résultat |
|---|---:|
| CER sur références, manqués = suppressions | 25.89% |
| WER sur références, manqués = suppressions | 32.94% |
| CER seulement sur régions appariées | 13.26% |
| WER seulement sur régions appariées | 22.45% |
| Précision détection conditionnelle aux régions annotées | 76.27% |
| Rappel des régions annotées | 78.40% |
| Exactitude des rôles géométriques sur paires appariées | 64.44% |

| Classe | CER avec manqués | WER avec manqués | Rappel localisation |
|---|---:|---:|---:|
| v_bar | 29.26% | 31.15% | 72.86% |
| h_bar | 3.48% | 9.02% | 76.92% |
| line | 43.79% | 38.10% | 78.64% |
| pie | 60.18% | 80.77% | 100.00% |

Attention aux secteurs : le rappel de localisation est excellent mais les
transcriptions de référence peuvent omettre le pourcentage inclus dans la même
ligne OCR, ce qui compte comme insertions. Le rôle prédit reste `unknown` pour
les tranches ambiguës ; il n'est pas remplacé par `other` simplement pour augmenter
le score. Les rôles des secteurs ont ici 0 % d'exactitude face aux rôles de
référence, ce qui expose la limite du parseur prudent.

Le protocole un-à-un pénalise aussi les axes verticaux fragmentés et les lignes
fusionnées ; les taux ne séparent pas totalement segmentation et reconnaissance.
Les transcriptions absentes des annotations sont exclues du CER/WER ; seules
les détections manquées de références transcrites sont comparées à une chaîne vide.

### Exemples inspectés

- Correct : `2009/10`, `2010/11`, `2011/12` sont identiques aux transcriptions
  dans `multi_col_101181.png`.
- Correct via CLI : `Haiti`, `Libya`, `Morocco`, et le titre du graphique sont
  détectés ; le titre contient toutefois `wasted;` au lieu de `wasted,`.
- Erreur de reconnaissance : `Legal & General (Unit Trust) Managers` devient
  `Legal & Ceneral (Unit Trust) Managers` ; `Funds value in billion GBP` devient
  `Funds value in billion CBP` dans `two_col_61697.png`.
- Segmentation/lecture difficile : `Number of participants` est apparié à
  `participants` seulement ; `Growth rate` devient `uroth`.
- Pour les secteurs, `Internet` est apparié à `Internet 355` : libellé et
  pourcentage ont été regroupés, avec une erreur sur le symbole `%`.
- Bruit réel d'annotation constaté visuellement : `two_col_6419.png`,
  `models[5].text_label` annonce `Children's IKEA`, alors que la boîte correspond
  au texte visible `IKEA food 5%`. Aucune correction du JSON officiel ni du score.

Exemples visuels : `images/21881727007008_boxes.png` (bonne localisation),
`images/two_col_3752_boxes.png` (axe vertical fragmenté), dans le lot final.
Les listes détaillées `good_examples`, `bad_examples` et `per_image` du fichier
`metrics.json` permettent de distinguer référence non appariée et lecture erronée.

### Tests exécutés

Rapport : `results/cv3/tests_20261117_013731_931165/test_report.json` ; **PASS**.

- 20 JSON et visualisations relus ; 540 textes et boîtes contrôlés.
- Images sources inchangées (SHA-256 avant/après OCR et relecture des exports).
- Boîtes finies dans les limites de l'image, scores entre 0 et 1, absence d'erreurs OCR bloquantes.
- Image blanche réellement passée dans EasyOCR : aucun texte, JSON et visualisation valides.
- Fichier PNG invalide rejeté ; fusion de doublons et conservation de répétitions spatiales testées.
- Références sans transcription laissées sans transcription ; calculs de distance vérifiés.
- Aucune attribution de rôle d'axe aux secteurs dans le lot final.
- Empreintes des scripts/documents CV1/CV2 et du modèle entraîné inchangées pendant les tests.
- Syntaxe des quatre nouveaux modules vérifiée avec `ast.parse`.

`results/cv3/final_report.json` rassemble le bilan mesuré.
Ces résultats sont une baseline exploitable pour poursuivre l'amélioration de
CV3, pas une garantie d'extraction numérique fiable. Aucun fichier précédent
modifié ou supprimé, aucun push GitHub. Arrêt après CV3, CV4 et NLP non commencés.
