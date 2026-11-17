"""Jointure des annotations officielles et images locales, audit et dédoublonnage."""
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from chart_classifier import ROOT, CLASSES
from prepare_cv2 import digest


def inspect(item):
    split, annotation = item
    try:
        data = json.loads(annotation.read_text(encoding='utf-8'))
        label = data.get('type')
        if label not in CLASSES:
            raise ValueError(f'type inconnu : {label}')
        source = ROOT / 'ChartQA Dataset' / split / 'png' / f'{annotation.stem}.png'
        key = digest(source)
        return {'image': source.relative_to(ROOT).as_posix(), 'split': split,
                'label': label, 'annotator': 'official:' + annotation.parent.parent.name,
                'pixel_sha256': key, 'annotation': str(annotation.relative_to(ROOT))}, None
    except (OSError, ValueError) as error:
        return None, {'annotation': str(annotation), 'reason': str(error)}


def main():
    directories = sorted((ROOT / 'annotations_cv2').glob('*/provenance.json'))
    if len(directories) != 1:
        raise ValueError('Une seule révision officielle attendue')
    folder = directories[0].parent
    items = [(s, p) for s in ('test', 'val', 'train') for p in sorted((folder / s).glob('*.json'))]
    valid, rejected = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (row, error) in enumerate(pool.map(inspect, items)):
            (valid if row else rejected).append(row or error)
            if index % 2000 == 0:
                print('Annotations vérifiées :', index, flush=True)
    # Écarter toutes les occurrences dont les vérités terrain sont contradictoires.
    labels_by_hash = {}
    for row in valid:
        labels_by_hash.setdefault(row['pixel_sha256'], set()).add(row['label'])
    kept, hashes, names = [], set(), set()
    for row in valid:  # priorité test, puis val : ne déplacer aucune image.
        key, name = row['pixel_sha256'], Path(row['image']).name
        if len(labels_by_hash[key]) > 1:
            rejected.append({**row, 'reason': 'conflicting_official_labels_for_identical_pixels'})
        elif key in hashes or name in names:
            rejected.append({**row, 'reason': 'duplicate_pixels_or_filename; heldout_priority'})
        else:
            kept.append(row)
            hashes.add(key)
            names.add(name)
    output = ROOT / 'results/cv2'
    fields = ['image', 'split', 'label', 'annotator']
    with (output / 'official_labels.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:row[k] for k in fields} for row in kept)
    report = {'source': json.loads(directories[0].read_text()), 'annotation_count': len(items),
              'matched_valid_before_dedup': len(valid), 'usable_images': len(kept),
              'raw_distribution': {s:dict(Counter(r['label'] for r in valid if r['split']==s)) for s in ('train','val','test')},
              'distribution': {s:{c:sum(r['split']==s and r['label']==c for r in kept) for c in CLASSES} for s in ('train','val','test')},
              'rejected': rejected,
              'limitation': 'Official types not exhaustively human-reviewed; near-duplicates with different names/pixels not detected.'}
    (output / 'official_annotation_quality.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'official_image_hashes.json').write_text(json.dumps({r['image']:r['pixel_sha256'] for r in kept}, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'rejected'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
