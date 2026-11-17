"""Évaluation exclusivement sur le split test d'un checkpoint supervisé."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from chart_classifier import ROOT, CLASSES, load_checkpoint, load_image
from prepare_cv2 import read_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        model, checkpoint = load_checkpoint(args.checkpoint)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Évaluation bloquée : {error}\n')
    # Revalide provenance et doublons de tout le manifeste d'entraînement.
    import csv
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        manifest = Path(temporary) / 'labels.csv'
        with manifest.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['image', 'split', 'label', 'annotator'])
            writer.writeheader()
            writer.writerows(checkpoint['rows'])
        rows, image_hashes = read_labels(manifest, return_hashes=True)
    if checkpoint.get('image_hashes') != image_hashes:
        raise ValueError('Images modifiées depuis l’entraînement, évaluation interrompue')
    tests = [r for r in rows if r['split'] == 'test']
    truth, predicted = [], []
    with torch.inference_mode():
        for row in tests:
            truth.append(CLASSES.index(row['label']))
            predicted.append(model(load_image(ROOT / row['image']).unsqueeze(0)).argmax(1).item())
    output = ROOT / 'results/cv2' / datetime.now().strftime('test_%Y%m%d_%H%M%S_%f')
    output.mkdir()
    metrics = {'split': 'test', 'n': len(tests), 'accuracy': accuracy_score(truth, predicted),
               'classification_report': classification_report(truth, predicted, labels=list(range(4)),
                    target_names=CLASSES, output_dict=True, zero_division=0)}
    metrics['checkpoint'] = str(args.checkpoint.resolve())
    metrics['selected_epoch'] = checkpoint['epoch']
    (output / 'predictions.json').write_text(json.dumps([
        {'image': row['image'], 'true_label': CLASSES[a], 'prediction': CLASSES[b]}
        for row, a, b in zip(tests, truth, predicted)], indent=2), encoding='utf-8')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    ConfusionMatrixDisplay.from_predictions(truth, predicted, labels=list(range(4)), display_labels=CLASSES)
    plt.tight_layout()
    plt.savefig(output / 'confusion_matrix.png')
    plt.close()
    for correct in (True, False):
        examples = [i for i, (a, b) in enumerate(zip(truth, predicted)) if (a == b) == correct][:6]
        if not examples:
            continue
        fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4), squeeze=False)
        for ax, i in zip(axes[0], examples):
            with Image.open(ROOT / tests[i]['image']) as image:
                ax.imshow(image.convert('RGB'))
            ax.set_title(f'Vrai : {CLASSES[truth[i]]}\nPrédit : {CLASSES[predicted[i]]}')
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(output / ('correct.png' if correct else 'incorrect.png'))
        plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f'Résultats test : {output}', flush=True)


if __name__ == '__main__':
    main()
