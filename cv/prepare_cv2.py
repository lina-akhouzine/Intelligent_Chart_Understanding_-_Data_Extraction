"""Audit des QA et préparation d'un CSV vierge pour annotation humaine."""
import csv
import hashlib
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageDraw
from chart_classifier import ROOT, CLASSES


def digest(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def read_labels(path, return_hashes=False):
    """Valide les labels sourcés (officiels ou humains) et les doublons exacts."""
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('CSV vide')
    seen_paths, seen_pixels = set(), {}
    for row in rows:
        if row['label'] not in CLASSES:
            raise ValueError(f"Étiquette absente/invalide : {row['image']} ; annoter humainement le CSV.")
        if row['split'] not in ('train', 'val', 'test') or not row['annotator'].strip():
            raise ValueError('Split ou provenance des étiquettes manquante')
        source = (ROOT / row['image']).resolve()
        if not source.is_relative_to((ROOT / 'ChartQA Dataset').resolve()):
            raise ValueError('Image hors dataset')
        if source.parent.name != 'png' or source.parent.parent.name != row['split']:
            raise ValueError('Le split doit correspondre au dossier original de l’image')
        if source in seen_paths:
            raise ValueError(f'Image répétée : {source}')
        seen_paths.add(source)
    # Décodage et hachage parallèles, sans modifier les fichiers sources.
    with ThreadPoolExecutor(max_workers=4) as pool:
        keys = list(pool.map(digest, [(ROOT / row['image']).resolve() for row in rows]))
    for row, key in zip(rows, keys):
        if key in seen_pixels:
            raise ValueError('Doublon de pixels détecté ; conserver une seule occurrence dans le CSV')
        seen_pixels[key] = row['split']
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != set(CLASSES):
            raise ValueError(f'{split} doit contenir les quatre classes ; compléter le lot annoté.')
    return (rows, {r['image']:key for r,key in zip(rows,keys)}) if return_hashes else rows


def main():
    out = ROOT / 'results' / 'cv2'
    out.mkdir(parents=True, exist_ok=True)
    report, selected = {}, []
    rng = random.Random(42)
    for split, count in [('train', 24), ('val', 8), ('test', 8)]:
        folder = ROOT / 'ChartQA Dataset' / split
        images = sorted((folder / 'png').glob('*.png'))
        qa = {}
        for path in sorted(folder.glob('*.json')):
            records = json.loads(path.read_text(encoding='utf-8'))
            missing = sorted({r['imgname'] for r in records if not (folder / 'png' / r['imgname']).is_file()})
            qa[path.name] = {'records': len(records), 'keys': sorted({k for r in records for k in r}),
                             'missing_images': missing}
        report[split] = {'images': len(images), 'qa': qa,
                         'annotation_folder_exists': (folder / 'annotations').is_dir()}
        selected.extend({'image': p.relative_to(ROOT).as_posix(), 'split': split,
                         'label': '', 'annotator': ''} for p in rng.sample(images, count))
    report['status'] = 'BLOCKED_NO_RELIABLE_CHART_TYPE_LABELS'
    report['class_distribution'] = {c: None for c in CLASSES}
    report['note'] = 'label des JSON QA = réponse à la question, jamais type du graphique.'
    (out / 'dataset_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    target = out / 'manual_labels.csv'
    # Ne jamais écraser des annotations déjà saisies.
    if not target.exists():
        with target.open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['image', 'split', 'label', 'annotator'])
            writer.writeheader()
            writer.writerows(selected)
        for page in range(2):
            sheet = Image.new('RGB', (1250, 5 * 240), 'white')
            draw = ImageDraw.Draw(sheet)
            for offset, row in enumerate(selected[page * 20:(page + 1) * 20]):
                x, y = offset % 4 * 312, offset // 4 * 240
                with Image.open(ROOT / row['image']) as image:
                    thumb = ImageOps.contain(image.convert('RGB'), (302, 200))
                sheet.paste(thumb, (x, y))
                draw.text((x + 3, y + 202), f"{page * 20 + offset + 1:02d} {row['split']} {Path(row['image']).stem}", fill='black')
            sheet.save(out / f'annotation_sheet_{page + 1}.png')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
