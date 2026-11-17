"""Tests techniques réels sans étiquettes ni métriques métier inventées."""
import json
from pathlib import Path
import torch
from chart_classifier import ROOT, CLASSES, build_model, load_image, load_checkpoint


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    output = ROOT / 'results/cv2'
    output.mkdir(parents=True, exist_ok=True)
    report = {'torch': torch.__version__, 'cuda': torch.cuda.is_available(),
              'status': 'TECHNICAL_ONLY_NO_SUPERVISED_TRAINING', 'test_metrics': None}
    # Le téléchargement éventuel est annoncé avant exécution ; pas de repli silencieux.
    model = build_model(pretrained=True).eval()
    rows = []
    import csv
    with (output / 'manual_labels.csv').open(encoding='utf-8') as handle:
        sample = list(csv.DictReader(handle))[:5]
    batch = torch.stack([load_image(ROOT / r['image']) for r in sample])
    assert batch.shape == (5, 3, 224, 224) and torch.isfinite(batch).all()
    with torch.inference_mode():
        before = model(batch)
    assert before.shape == (5, 4) and torch.isfinite(before).all()
    checkpoint = ROOT / 'models_cv2' / 'smoke_untrained.pt'
    torch.save({'state_dict': model.state_dict(), 'classes': CLASSES, 'trained': False,
                'note': 'ImageNet backbone, random head; NOT a trained chart classifier'}, checkpoint)
    restored = build_model(pretrained=False).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)['state_dict'])
    with torch.inference_mode():
        after = restored(batch)
    assert torch.equal(before, after)
    try:
        load_checkpoint(checkpoint)
    except ValueError:
        report['untrained_prediction_guard'] = 'PASS'
    else:
        raise AssertionError('Checkpoint non entraîné accepté')
    for row, logits in zip(sample, after):
        rows.append({'image': row['image'], 'raw_logits_untrained_head': logits.tolist(),
                     'warning': 'Pas une prédiction valide de type de graphique'})
    report.update({'pretrained_weights': 'IMAGENET1K_V1', 'images_tested': 5,
                   'input_shape': list(batch.shape), 'output_shape': list(after.shape),
                   'checkpoint_roundtrip_exact': True, 'forward_passes': rows})
    (output / 'technical_test.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
