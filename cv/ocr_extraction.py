"""CV3 : OCR des textes visibles, sans extraction des valeurs de barres/courbes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from preprocessing import preprocess
from chart_text_parser import annotation_references, estimate_roles, iou

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'models_cv2/resnet18_20261117_010609_069660.pt'


def create_reader(download=False):
    import torch
    import easyocr
    torch.set_num_threads(4)
    directory = ROOT / 'models_cv3/easyocr'
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available(),
                          model_storage_directory=str(directory / 'weights'),
                          user_network_directory=str(directory / 'user_network'),
                          download_enabled=download, verbose=False)


def deduplicate(detections):
    """Fusion spatiale ; les hypothèses concurrentes restent dans alternatives."""
    kept = []
    for item in sorted(detections, key=lambda d:d['confidence'], reverse=True):
        duplicate = None
        for existing in kept:
            overlap = iou(item['bbox'], existing['bbox'])
            similarity = SequenceMatcher(None, item['text'].casefold(), existing['text'].casefold()).ratio()
            if overlap >= .75 or (overlap >= .45 and similarity >= .55):
                duplicate = existing
                break
        if duplicate is None:
            kept.append({**item, 'variants': [item['variant']], 'alternatives': []})
        else:
            if item['variant'] not in duplicate['variants']:
                duplicate['variants'].append(item['variant'])
            duplicate['alternatives'].append({'text': item['text'], 'confidence': item['confidence'], 'variant': item['variant']})
    return sorted(kept, key=lambda d:(d['bbox'][1], d['bbox'][0]))


class OCRExtractor:
    def __init__(self, reader=None, checkpoint=DEFAULT_CHECKPOINT, download=False):
        self.reader = reader if reader is not None else create_reader(download)
        self.classifier, self.classes, self.classifier_error = None, None, None
        if checkpoint is not None:
            try:
                from chart_classifier import load_checkpoint
                self.classifier, metadata = load_checkpoint(checkpoint)
                self.classes = metadata['classes']
            except (OSError, ValueError) as error:
                self.classifier_error = str(error)

    def extract(self, path, variants=('original', 'clahe'), annotation=None):
        path = Path(path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        versions = preprocess(path)  # CV1, lecture seule, dimensions natives.
        height, width = versions['gray'].shape
        start = time.monotonic()
        errors, warnings, detections = [], [], []
        chart_type, chart_probabilities = None, None
        if self.classifier is not None:
            import torch
            from chart_classifier import load_image
            with torch.inference_mode():
                probabilities = self.classifier(load_image(path).unsqueeze(0)).softmax(1)[0]
            chart_type = self.classes[probabilities.argmax().item()]
            chart_probabilities = dict(zip(self.classes, probabilities.tolist()))
        elif self.classifier_error:
            warnings.append('CV2 indisponible : ' + self.classifier_error)
        for variant in variants:
            if variant not in versions:
                raise ValueError(f'Variante inconnue : {variant}')
            pixels = cv2.cvtColor(versions[variant], cv2.COLOR_RGB2BGR) if variant == 'original' else versions[variant]
            try:
                raw = self.reader.readtext(pixels, detail=1, paragraph=False, workers=0,
                                           rotation_info=[90, 270], min_size=5,
                                           mag_ratio=1.5, canvas_size=2560)
                for polygon, text, confidence in raw:
                    confidence = float(confidence)
                    if not str(text).strip() or not math.isfinite(confidence):
                        continue
                    points = [[float(np.clip(x, 0, width)), float(np.clip(y, 0, height))] for x,y in polygon]
                    box = [min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)]
                    if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                        warnings.append(f'{variant}: boîte invalide ignorée')
                        continue
                    detections.append({'text': str(text).strip(), 'bbox': box, 'polygon': points,
                                       'confidence': max(0., min(1., confidence)), 'variant': variant})
            except (RuntimeError, ValueError, cv2.error) as error:
                errors.append(f'OCR {variant}: {error}')
        texts = deduplicate(detections)
        references = None
        if annotation is not None:
            references, issues = annotation_references(annotation)
            warnings.extend(issues)
        estimate_roles(texts, width, height, chart_type, references)
        if not texts:
            warnings.append('Aucun texte détecté ; cela ne prouve pas l’absence de texte.')
        if any(t['confidence'] < .5 for t in texts):
            warnings.append('Certains textes ont une confiance OCR inférieure à 0.5.')
        if any(any(a['text'].casefold() != t['text'].casefold() for a in t['alternatives']) for t in texts):
            warnings.append('Hypothèses concurrentes entre variantes : consulter alternatives.')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        result = {'schema_version': 1, 'image': path.name, 'source': str(path.resolve()),
                  'source_sha256': before, 'width': width, 'height': height,
                  'engine': 'EasyOCR 1.7.2 / en', 'chart_type': chart_type,
                  'chart_type_source': 'CV2 checkpoint prediction' if chart_type else None,
                  'chart_probabilities': chart_probabilities, 'variants': list(variants),
                  'texts': texts, 'errors': errors, 'uncertainties': warnings,
                  'seconds': time.monotonic()-start,
                  'scope': 'Visible text only. Numeric OCR strings are NOT extracted chart data.',
                  'annotation_assistance': 'regions/roles only; reference text never fed into OCR' if annotation else None}
        return result, versions['original']


def save_result(result, original, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (Path(result['image']).stem + '.json')
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    image = Image.fromarray(original)
    draw = ImageDraw.Draw(image)
    for index, text in enumerate(result['texts']):
        color = 'orange' if text['role'] == 'unknown' else 'red'
        draw.line([tuple(p) for p in text['polygon']] + [tuple(text['polygon'][0])], fill=color, width=2)
        draw.text((text['bbox'][0], max(0, text['bbox'][1]-11)), str(index), fill='blue')
    image.save(folder / (Path(result['image']).stem + '_boxes.png'))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path, nargs='+')
    parser.add_argument('--variants', nargs='+', default=['original', 'clahe'])
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--annotation', type=Path, help='Assistance de régions, jamais transcription injectée')
    parser.add_argument('--download-models', action='store_true')
    args = parser.parse_args()
    annotation = json.loads(args.annotation.read_text(encoding='utf-8')) if args.annotation else None
    if annotation and len(args.image) != 1:
        parser.error('--annotation requiert une seule image')
    extractor = OCRExtractor(checkpoint=args.checkpoint, download=args.download_models)
    destination = ROOT / 'results/cv3' / datetime.now().strftime('ocr_%Y%m%d_%H%M%S_%f')
    for path in args.image:
        result, original = extractor.extract(path, args.variants, annotation)
        target = save_result(result, original, destination)
        print(f"{path.name}: {len(result['texts'])} textes -> {target}", flush=True)


if __name__ == '__main__':
    main()
