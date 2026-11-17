"""Évaluation OCR sur références partielles, appariement géométrique indépendant du texte."""
from __future__ import annotations
import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from chart_text_parser import annotation_references, estimate_roles, iou
from ocr_extraction import ROOT, OCRExtractor, save_result


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def match_boxes(refs, detections, threshold=.3):
    if not refs or not detections:
        return []
    overlaps = np.array([[iou(r['bbox'], d['bbox']) for d in detections] for r in refs])
    # Maximiser d'abord le nombre de correspondances admissibles puis leur IoU.
    costs = (overlaps >= threshold).astype(float) * (min(overlaps.shape)+1) + overlaps
    rows, columns = linear_sum_assignment(costs, maximize=True)
    return [(int(r), int(c), float(overlaps[r,c])) for r,c in zip(rows,columns) if overlaps[r,c] >= threshold]


def score_image(refs, detections, width, height):
    # Les boîtes hors image ne constituent pas des références évaluables.
    refs = [r for r in refs if 0 <= r['bbox'][0] < r['bbox'][2] <= width and
            0 <= r['bbox'][1] < r['bbox'][3] <= height]
    matches = match_boxes(refs, detections)
    paired = {r:d for r,d,_ in matches}
    eligible = [i for i,d in enumerate(detections) if any(iou(d['bbox'],r['bbox']) >= .1 for r in refs)]
    counts = {'reference_regions': len(refs), 'detections': len(detections),
              'detections_in_annotated_regions': len(eligible), 'matched_regions': len(matches),
              'transcribed_references': 0, 'matched_transcribed_references': 0,
              'characters': 0, 'words': 0, 'character_edits': 0, 'word_edits': 0,
              'matched_characters': 0, 'matched_words': 0, 'matched_character_edits': 0, 'matched_word_edits': 0,
              'role_correct_geometry': 0}
    examples, roles = [], Counter()
    for r,d,_ in matches:
        correct = refs[r]['role'] == detections[d]['role']
        counts['role_correct_geometry'] += int(correct)
        roles[(refs[r]['role'], detections[d]['role'])] += 1
    for index, reference in enumerate(refs):
        if reference['text'] is None:
            continue
        truth = normalize(reference['text'])
        if not truth:
            continue
        detection = detections[paired[index]] if index in paired else None
        predicted = normalize(detection['text']) if detection else ''
        char_edits = edit_distance(truth, predicted)
        word_edits = edit_distance(truth.split(), predicted.split())
        counts['transcribed_references'] += 1
        counts['characters'] += len(truth)
        counts['words'] += len(truth.split())
        counts['character_edits'] += char_edits
        counts['word_edits'] += word_edits
        if detection is not None:
            counts['matched_transcribed_references'] += 1
            counts['matched_characters'] += len(truth)
            counts['matched_words'] += len(truth.split())
            counts['matched_character_edits'] += char_edits
            counts['matched_word_edits'] += word_edits
        examples.append({'reference': reference['text'], 'ocr': detection['text'] if detection else None,
                         'reference_role': reference['role'], 'reference_field': reference['field'],
                         'bbox': reference['bbox'], 'matched': detection is not None,
                         'character_edits': char_edits, 'word_edits': word_edits})
    return counts, examples, [{'reference_role': a, 'predicted_role': b, 'count':n} for (a,b),n in roles.items()]


def derive(counts):
    def ratio(a,b):
        return counts[a]/counts[b] if counts[b] else None
    return {'CER_reference_regions_misses_as_deletions': ratio('character_edits','characters'),
            'WER_reference_regions_misses_as_deletions': ratio('word_edits','words'),
            'CER_matched_only': ratio('matched_character_edits','matched_characters'),
            'WER_matched_only': ratio('matched_word_edits','matched_words'),
            'detection_precision_annotated_regions': ratio('matched_regions','detections_in_annotated_regions'),
            'detection_recall_reference_regions': ratio('matched_regions','reference_regions'),
            'role_accuracy_geometry_on_matches': ratio('role_correct_geometry','matched_regions')}


def audit_one(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    refs, issues = annotation_references(data)
    fields = Counter()
    def walk(value, prefix=''):
        if isinstance(value, dict):
            for key,item in value.items():
                name = f'{prefix}.{key}' if prefix else key
                fields[name] += 1
                if key not in ('bbox','bboxes'):
                    walk(item,name)
    walk(data.get('general_figure_info', {}))
    return {'file': str(path.relative_to(ROOT)), 'type': data.get('type'),
            'fields': fields, 'regions': len(refs), 'transcribed_regions': sum(r['text'] is not None for r in refs),
            'roles': dict(Counter(r['role'] for r in refs)), 'issues': issues}


def audit_annotations():
    root = next((ROOT / 'annotations_cv2').glob('*/provenance.json')).parent
    paths = [p for s in ('train','val','test') for p in sorted((root/s).glob('*.json'))]
    fields, roles = Counter(), Counter()
    total_regions, transcribed, files_with_transcripts = 0, 0, 0
    issues = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index,item in enumerate(pool.map(audit_one, paths)):
            fields.update(item['fields']); roles.update(item['roles'])
            total_regions += item['regions']; transcribed += item['transcribed_regions']
            files_with_transcripts += int(item['transcribed_regions'] > 0)
            if item['issues']:
                issues.append({'file': item['file'], 'issues': item['issues']})
            if index % 4000 == 0:
                print(f'Audit annotations : {index}/{len(paths)}', flush=True)
    report = {'revision': root.name, 'annotation_files': len(paths), 'field_presence': dict(fields),
              'reference_regions': total_regions, 'transcribed_regions': transcribed,
              'files_with_transcribed_regions': files_with_transcripts, 'reference_roles': dict(roles),
              'issues_by_file': issues,
              'exclusions': 'models.x/y/value/name never treated as OCR text ground truth; missing text not invented'}
    destination = ROOT / 'results/cv3'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'annotation_audit.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'issues_by_file'},indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--per-class', type=int, default=5)
    parser.add_argument('--variants', nargs='+', default=['original','clahe'])
    parser.add_argument('--reuse-ocr', type=Path, help='Recalculer les rôles/métriques depuis un lot OCR enregistré')
    args = parser.parse_args()
    if args.audit_only:
        audit_annotations()
        return
    if args.per_class < 1:
        parser.error('--per-class doit être positif')
    if args.reuse_ocr:
        selected = json.loads((args.reuse_ocr/'sample_manifest.json').read_text(encoding='utf-8'))
    else:
        with (ROOT/'results/cv2/official_labels.csv').open(encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        rng = random.Random(43)
        selected = [r for c in ('v_bar','h_bar','line','pie') for r in
                    rng.sample([r for r in rows if r['split']=='test' and r['label']==c], args.per_class)]
    root = next((ROOT/'annotations_cv2').glob('*/provenance.json')).parent
    destination = ROOT/'results/cv3'/datetime.now().strftime('evaluation_%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True)
    (destination/'sample_manifest.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    extractor = None if args.reuse_ocr else OCRExtractor()
    totals, by_class, records, all_examples = Counter(), {}, [], []
    for index,row in enumerate(selected):
        path = ROOT/row['image']
        annotation_path = root/row['split']/(path.stem+'.json')
        annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
        if args.reuse_ocr:
            import hashlib
            from PIL import Image
            result = json.loads((args.reuse_ocr/'images'/(path.stem+'.json')).read_text(encoding='utf-8'))
            if hashlib.sha256(path.read_bytes()).hexdigest() != result['source_sha256']:
                raise ValueError('Source OCR modifiée')
            refs, _ = annotation_references(annotation)
            estimate_roles(result['texts'],result['width'],result['height'],result['chart_type'],refs)
            result['roles_recomputed_from'] = str(args.reuse_ocr)
            with Image.open(path) as image:
                original = np.asarray(image.convert('RGB'))
        else:
            result, original = extractor.extract(path, args.variants, annotation)
        save_result(result,original,destination/'images')
        refs, issues = annotation_references(annotation)
        counts, examples, role_matrix = score_image(refs,result['texts'],result['width'],result['height'])
        totals.update(counts)
        by_class.setdefault(row['label'],Counter()).update(counts)
        records.append({'image': path.name, 'class_reference': row['label'], 'counts': counts,
                        'metrics': derive(counts), 'examples': examples, 'role_confusion': role_matrix,
                        'ocr_errors': result['errors'], 'annotation_issues': issues})
        all_examples.extend({'image': path.name, **example} for example in examples)
        print(f"{index+1}/{len(selected)} {row['label']} {path.name}: {len(result['texts'])} textes, {counts['matched_regions']}/{counts['reference_regions']} régions appariées",flush=True)
    report = {'images': len(selected), 'sampling': 'seed 43, test officiel, 5 images/classe par défaut',
              'reused_ocr_run': str(args.reuse_ocr) if args.reuse_ocr else None,
              'counts': dict(totals), 'metrics': derive(totals),
              'by_class': {c:{'counts':dict(v),'metrics':derive(v)} for c,v in by_class.items()},
              'protocol': {'matching': 'Hungarian one-to-one, IoU >= 0.30, no reference text used',
                           'precision_denominator': 'OCR detections overlapping a known reference region at IoU >= 0.10 only',
                           'normalization': 'Unicode NFKC, casefold, whitespace collapsed; punctuation retained',
                           'missing_text': 'regions without transcripts evaluated only for localization and roles',
                           'roles': 'geometry only scored; annotation assistance not scored',
                           'CER_WER': 'micro-average; misses count as deletions; extra unannotated text excluded',
                           'warning': 'Partial annotation evaluation, NOT full-page OCR precision/recall or role accuracy.'},
              'per_image': records,
              'good_examples': [e for e in all_examples if e['matched'] and e['character_edits']==0][:15],
              'bad_examples': sorted([e for e in all_examples if e['character_edits']>0],key=lambda e:e['character_edits'],reverse=True)[:15]}
    (destination/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'directory':str(destination),'images':len(selected),'metrics':report['metrics']},indent=2),flush=True)


if __name__ == '__main__':
    main()
