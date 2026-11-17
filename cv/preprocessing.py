"""CV1 : prétraitement ChartQA, sans modification des images sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def discover_images(dataset: Path) -> list[Path]:
    """Recherche déterministe dans les dossiers PNG du dataset uniquement."""
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset introuvable : {dataset}")
    return sorted(p for p in dataset.rglob('*') if p.suffix.lower() == '.png')


def preprocess(path: Path) -> dict[str, np.ndarray]:
    """Conserve le RGB original et retourne les étapes, sans redimensionnement."""
    with Image.open(path) as image:
        image.verify()
    # imdecode/fromfile prend en charge les chemins Windows avec accents.
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or min(bgr.shape[:2]) < 3:
        raise ValueError(f"Image invalide ou trop petite : {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    # Filtre bilatéral modéré : réduit le bruit tout en préservant les contours.
    denoised = cv2.bilateralFilter(contrast, 5, 20, 20)
    block = min(31, min(gray.shape) if min(gray.shape) % 2 else min(gray.shape) - 1)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, 11,
    )
    return {'original': cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            'gray': gray, 'clahe': contrast, 'denoised': denoised, 'binary': binary}


def save_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def run(images: list[Path], output: Path, show: bool = True) -> Path:
    """Sauvegarde chaque étape, une comparaison par image et une vue d'ensemble."""
    if not images:
        raise ValueError('Aucune image PNG trouvée.')
    output.mkdir(parents=True, exist_ok=True)
    # Un nouveau sous-dossier à chaque exécution évite tout écrasement.
    from datetime import datetime
    destination = output / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    destination.mkdir()
    overview, axes = plt.subplots(len(images), 4, figsize=(18, 4 * len(images)), squeeze=False)
    records = []
    stages = [('original', 'Originale couleur'), ('gray', 'Niveaux de gris'),
              ('clahe', 'Contraste CLAHE'), ('binary', 'Binarisation adaptative')]
    try:
        for index, path in enumerate(images):
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            versions = preprocess(path)
            height, width = versions['gray'].shape
            print(f'{path.name} : largeur={width}, hauteur={height}, resolution={width}x{height} pixels')
            folder = destination / f'{index + 1:02d}_{path.stem}'
            folder.mkdir()
            for name, array in versions.items():
                save_image(folder / f'{name}.png', array)
                with Image.open(folder / f'{name}.png') as saved:
                    assert saved.size == (width, height)
                    assert np.array_equal(np.asarray(saved), array)
            assert set(np.unique(versions['binary'])).issubset({0, 255})
            fig, row = plt.subplots(1, 4, figsize=(18, 5))
            for col, (name, title) in enumerate(stages):
                for ax in (row[col], axes[index, col]):
                    ax.imshow(versions[name], cmap=None if name == 'original' else 'gray', vmin=0, vmax=255)
                    ax.set_title(title)
                    ax.axis('off')
            fig.suptitle(f'{path.name} — {width} × {height} px')
            fig.tight_layout()
            fig.savefig(folder / 'comparison.png', dpi=160)
            plt.close(fig)
            assert before == hashlib.sha256(path.read_bytes()).hexdigest(), 'Source modifiée'
            records.append({'source': str(path.resolve()), 'sha256': before,
                            'width': width, 'height': height, 'output': str(folder.resolve()),
                            'binary_black_fraction': float(np.mean(versions['binary'] == 0)),
                            'checks': 'PNG reload, dimensions, pixel equality, binary values, source hash: OK'})
        overview.tight_layout()
        overview.savefig(destination / 'comparison_all.png', dpi=160)
        (destination / 'manifest.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'OK : {len(records)} images. Resultats : {destination}')
        if show:
            plt.show()
    finally:
        plt.close(overview)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'ChartQA Dataset')
    parser.add_argument('--image', type=Path, help='Une image explicite au lieu de la sélection automatique')
    parser.add_argument('--count', type=int, default=1, help='Nombre de graphiques répartis dans le dataset')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'preprocessed')
    parser.add_argument('--no-show', action='store_true', help='Sauvegarder sans ouvrir de fenêtre')
    args = parser.parse_args()
    try:
        if args.count < 1:
            raise ValueError('--count doit être positif')
        candidates = [args.image] if args.image else discover_images(args.dataset)
        if not candidates:
            raise ValueError('Aucune image PNG trouvée')
        indices = np.linspace(0, len(candidates) - 1, min(args.count, len(candidates)), dtype=int)
        run([candidates[int(i)] for i in indices], args.output, show=not args.no_show)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f'Erreur CV1 : {error}\n')


if __name__ == '__main__':
    main()
