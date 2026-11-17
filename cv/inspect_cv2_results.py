"""Illustrer les sorties réelles et une erreur de validation sans inventer d'erreur test."""
import json
import textwrap
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from chart_classifier import ROOT, CLASSES, load_checkpoint, load_image


def draw(rows, destination, title):
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 5), squeeze=False)
    for ax, row in zip(axes[0], rows):
        with Image.open(ROOT / row['image']) as image:
            ax.imshow(image.convert('RGB'))
        ax.set_title(f"{row['split']} : vrai={row['true_label']}\nprédit={row['prediction']}")
        ax.axis('off')
    fig.suptitle(textwrap.fill(title, width=45 if len(rows) == 1 else 120), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    torch.set_num_threads(4)
    results = ROOT / 'results/cv2'
    metrics_path = sorted(results.glob('test_*/metrics.json'))[-1]
    metrics = json.loads(metrics_path.read_text())
    model, checkpoint = load_checkpoint(metrics['checkpoint'])
    predictions = json.loads((metrics_path.parent / 'predictions.json').read_text())
    assert len(predictions) == metrics['n'] == 1509
    assert sum(r['prediction'] == r['true_label'] for r in predictions) == 1509
    examples = [{**next(r for r in predictions if r['true_label'] == c), 'split': 'test'} for c in CLASSES]
    # Recalcul direct sur une image de chaque classe, après export des résultats.
    with torch.inference_mode():
        for row in examples:
            predicted = CLASSES[model(load_image(ROOT / row['image']).unsqueeze(0)).argmax(1).item()]
            assert predicted == row['prediction']
    draw(examples, metrics_path.parent / 'examples_by_class.png', 'Exemples réels du test — une image par classe')
    validation = [r for r in checkpoint['rows'] if r['split'] == 'val']
    caches = [torch.load(p, weights_only=True) for p in results.glob('features_val_*.pt')]
    cache = next(c for c in caches if len(c['labels']) == len(validation))
    assert cache['labels'].tolist() == [CLASSES.index(r['label']) for r in validation]
    with torch.inference_mode():
        predicted = model.fc(cache['features']).argmax(1).tolist()
    incorrect = []
    for row, prediction in zip(validation, predicted):
        if CLASSES[prediction] != row['label']:
            with torch.inference_mode():
                direct = model(load_image(ROOT / row['image']).unsqueeze(0)).argmax(1).item()
            assert prediction == direct
            incorrect.append({'image': row['image'], 'split': 'val', 'true_label': row['label'],
                              'prediction': CLASSES[prediction]})
    if incorrect:
        draw(incorrect[:4], metrics_path.parent / 'validation_incorrect.png',
             'Erreurs de VALIDATION — ces images ne font pas partie du test')
    report = {'test_correct': 1509, 'test_incorrect': 0,
              'test_direct_forward_rechecks': examples, 'validation_incorrect': incorrect,
              'note': 'No incorrect test examples exist; validation errors shown separately.'}
    (metrics_path.parent / 'prediction_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
