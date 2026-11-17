"""Rôles géométriques et lecture prudente des annotations textuelles officielles."""
from __future__ import annotations
import math
import re

ROLES = ['title', 'x_axis_title', 'y_axis_title', 'x_category', 'y_category',
         'x_tick', 'y_tick', 'legend', 'other', 'unknown']


def bbox_xyxy(value):
    """Convertit x/y/w/h ou une liste de boîtes en enveloppe x1/y1/x2/y2."""
    if isinstance(value, list):
        boxes = [b for item in value if (b := bbox_xyxy(item)) is not None]
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)] if boxes else None
    if not isinstance(value, dict) or not all(k in value for k in ('x', 'y', 'w', 'h')):
        return None
    try:
        x, y, w, h = [float(value[k]) for k in ('x', 'y', 'w', 'h')]
        return [x, y, x+w, y+h] if all(map(math.isfinite, (x,y,w,h))) and w > 0 and h > 0 else None
    except (TypeError, ValueError):
        return None


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def annotation_references(data):
    """N'utilise jamais models.x/y/value comme transcription de texte visible."""
    general = data.get('general_figure_info', {})
    refs, issues = [], []

    def add(text, box, role, field):
        bounds = bbox_xyxy(box)
        if bounds is None:
            if text is not None:
                issues.append(f'{field}: transcription sans boîte exploitable')
            return
        if isinstance(text, (str, int, float)):
            text = str(text).strip() or None
        else:
            text = None
        refs.append({'text': text, 'bbox': bounds, 'role': role, 'field': field})

    title = general.get('title', {})
    if isinstance(title, dict):
        add(title.get('text'), title.get('bbox'), 'title', 'general_figure_info.title')
    for name in ('x_axis', 'y_axis'):
        axis = general.get(name, {})
        if not isinstance(axis, dict):
            continue
        # ChartQA utilise souvent x pour l'axe catégoriel d'un h_bar (physiquement Y).
        physical = ('y' if name == 'x_axis' else 'x') if data.get('type') == 'h_bar' else name[0]
        label = axis.get('label', {})
        if isinstance(label, dict):
            add(label.get('text'), label.get('bbox'), f'{physical}_axis_title', f'{name}.label')
        major = axis.get('major_labels', {})
        if isinstance(major, dict):
            boxes = major.get('bboxes', [])
            values = major.get('values', [])
            if not isinstance(boxes, list):
                issues.append(f'{name}.major_labels.bboxes: format non pris en charge')
                continue
            if not isinstance(values, list):
                values = []
            if len(boxes) != len(values):
                issues.append(f'{name}.major_labels: {len(boxes)} boîtes, {len(values)} valeurs ; aucune valeur complétée')
            role = f'{physical}_category' if name == 'x_axis' else f'{physical}_tick'
            for index, box in enumerate(boxes):
                add(values[index] if len(values) == len(boxes) else None, box, role, f'{name}.major_labels[{index}]')
    legend = general.get('legend', {})
    if isinstance(legend, dict):
        for index, item in enumerate(legend.get('items', [])):
            if isinstance(item, dict) and isinstance(item.get('label'), dict):
                label = item['label']
                add(label.get('text'), label.get('bbox'), 'legend', f'legend.items[{index}].label')
    # Certains diagrammes à secteurs ont des textes de tranches explicitement annotés.
    for index, model in enumerate(data.get('models', [])):
        if isinstance(model, dict) and 'text_label' in model and 'text_bbox' in model:
            add(model['text_label'], model['text_bbox'], 'other', f'models[{index}].text_label')
    return refs, issues


def numeric(text):
    return bool(re.fullmatch(r'[+−–-]?\s*[\d\s,.]+\s*[%$€£kKmMbB]*', text.strip()))


def estimate_roles(detections, width, height, chart_type=None, references=None):
    """Heuristiques sans texte de référence. Assistance annotation exportée séparément."""
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        x, y = (x1+x2)/2/width, (y1+y2)/2/height
        bw, bh = (x2-x1)/width, (y2-y1)/height
        text = detection['text']
        role, certainty, reason = 'unknown', .25, 'position ambiguë'
        if y > .94:
            role, certainty, reason = 'other', .6, 'pied de figure probable'
        elif y < .14 and bw > .22 and not numeric(text):
            role, certainty, reason = 'title', .6, 'texte large en haut'
        elif chart_type == 'pie':
            role, certainty, reason = 'unknown', .25, 'secteur sans axes : tranche ou légende indéterminée'
        elif x < .07 and bh > .12:
            role, certainty, reason = 'y_axis_title', .65, 'texte vertical en marge gauche'
        elif .78 < y < .93 and bw > .2 and not numeric(text):
            role, certainty, reason = 'x_axis_title', .45, 'texte inférieur large ; légende possible'
        elif .65 < y < .9 and .1 < x < .95 and numeric(text):
            role, certainty, reason = 'x_tick' if chart_type == 'h_bar' else 'x_category', .45, 'texte numérique sous le tracé probable'
        elif x < .18 and .14 < y < .8:
            role = 'y_category' if chart_type == 'h_bar' else 'y_tick' if numeric(text) else 'unknown'
            certainty, reason = .5, 'marge gauche du tracé probable'
        detection.update(role=role, role_confidence=certainty, role_source='geometry', role_reason=reason)
        if references is not None:
            ranked = sorted(((iou(detection['bbox'], r['bbox']), r) for r in references), key=lambda p:p[0], reverse=True)
            score, reference = ranked[0] if ranked else (0., None)
            detection['annotation_assistance'] = ({'role': reference['role'], 'iou': score,
                                                   'field': reference['field'], 'source': 'official_region'}
                                                  if score >= .3 else None)
    return detections
