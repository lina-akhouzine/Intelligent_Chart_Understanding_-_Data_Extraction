# Chart2Data Intelligence — CV1

Le prétraitement prépare les images ChartQA à un futur OCR. Il ne réalise ni OCR,
ni extraction des données, ni classification CV2.

## Environnement

Python 3.13.5 sous Windows. `.venv` créée avec `--system-site-packages` pour
réutiliser les dépendances présentes sans téléchargement ni réinstallation :
opencv-python 4.13.0.92, numpy 2.3.0, pandas 2.3.3, matplotlib 3.10.7,
Pillow 11.2.1. Tous les imports ont réussi dans la `.venv`.
Cette configuration dépend des paquets du Python de base et n'est pas totalement
isolée. Pandas est vérifié pour la suite du projet, mais inutile au pipeline CV1.
Le nouveau `.vscode/settings.json` indique l'interpréteur par défaut ; si VS Code
a déjà mémorisé un interpréteur, sélectionner `.venv/Scripts/python.exe` avec
« Python: Select Interpreter ».

## Opérations et sorties

Découverte récursive des PNG dans `ChartQA Dataset`, sélection déterministe
répartie sur la liste triée, validation Pillow/OpenCV, dimensions en pixels,
conversion en gris, CLAHE (clip 2, grille 8×8), filtre bilatéral (5, 20, 20),
seuillage adaptatif gaussien (fenêtre 31, C=11 ; fenêtre réduite si nécessaire).
Les chemins par défaut sont relatifs au script, indépendamment du terminal.

Chaque exécution crée `results/preprocessed/run_<date>_<heure>_<microsecondes>/` :

- Par image : `original.png` (copie couleur RGB décodée), `gray.png`, `clahe.png`,
  `denoised.png`, `binary.png` et `comparison.png`.
- `comparison_all.png` : les quatre versions demandées pour toutes les images.
- `manifest.json` : sources, SHA-256, dimensions, sorties et vérifications.

Les sources ne sont jamais écrites. La copie RGB ne préserve pas nécessairement
les métadonnées ou la transparence de la source ; le fichier source reste intact.

## Commandes PowerShell

Depuis la racine du projet, commandes réellement exécutées :

```powershell
python --version
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -c "import sys,cv2,numpy,pandas,matplotlib,PIL; print(sys.executable); print('Imports OK')"
$env:MPLBACKEND='Agg'
.\.venv\Scripts\python.exe cv/preprocessing.py --count 5 --no-show
.\.venv\Scripts\python.exe cv/preprocessing.py --no-show
.\.venv\Scripts\python.exe cv/preprocessing.py --image cv/introuvable.png --no-show
```

Le dernier appel est un test négatif volontaire : code de sortie 1 avec message
explicite pour fichier absent. Les deux traitements précédents réussissent.
Les commandes ont été exécutées dans le terminal de l'agent, **pas dans le
terminal intégré de VS Code** : l'outil d'automatisation Windows disponible
interdit l'exécution de commandes terminal par l'interface graphique.

Pour relancer dans le terminal intégré de VS Code avec affichage interactif :

```powershell
# Dans un terminal neuf, sans MPLBACKEND=Agg :
.\.venv\Scripts\python.exe cv/preprocessing.py
.\.venv\Scripts\python.exe cv/preprocessing.py --count 5
# Une image particulière :
.\.venv\Scripts\python.exe cv/preprocessing.py --image "ChartQA Dataset/test/png/00339007006077.png"
```

L'activation du venv n'est pas nécessaire avec son exécutable explicite.
L'affichage interactif est implémenté avec `plt.show()` ; les tests automatisés
ont utilisé le mode sans fenêtre et la figure enregistrée a été inspectée.

## Résultats des essais

Dataset détecté : train 18 317 PNG, val 1 056, test 1 509 (total 20 882).
Lot principal : `results/preprocessed/run_20261117_002336_247346/`.

| Image | Dimensions | Observation visuelle sur la comparaison |
|---|---|---|
| 00339007006077 | 850×600 | Pays, pourcentages et axe lisibles ; petites notes épaissies |
| multi_col_60554 | 800×557 | Valeurs, catégories et légende 2015/2016 lisibles ; couleurs perdues |
| two_col_22358 | 800×557 | Années inclinées et valeurs conservées ; barres évidées |
| two_col_5693 | 800×776 | Régions et graduations lisibles ; petits caractères plus épais |
| two_col_978 | 800×607 | Courbe et axes lisibles ; graduations inclinées plus dentelées |

Les 25 PNG d'étapes ont été rechargés : dimensions et pixels identiques aux
tableaux calculés. Valeurs binaires limitées à 0/255 et SHA-256 des cinq sources
inchangé avant/après. Cinq comparaisons individuelles et une globale : 31 PNG
dans le lot principal, plus un manifeste JSON. Un second lancement par défaut
sur une image a également réussi.

## Limites

Contrôle de lisibilité qualitatif sur cinq graphiques seulement, principalement
des barres et une courbe ; aucune mesure de précision OCR ni garantie sur tout
le dataset. La binarisation épaissit le texte, accentue certaines grilles et
icônes de capture, transforme les aplats en contours et supprime la distinction
des couleurs des légendes. Les gris/CLAHE et l'original restent disponibles pour
choisir ultérieurement l'entrée OCR adaptée. Aucun redressement, agrandissement
ou retrait d'interface n'est appliqué. La résolution affichée est en pixels,
pas une résolution d'impression en DPI. CV2 attend l'autorisation utilisateur.

Aucun fichier initial, notebook, modèle ou élément du dataset n'a été modifié,
supprimé ou déplacé. Aucun push GitHub.
