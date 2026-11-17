"""Transfert d'apprentissage supervisé ; refuse les labels absents."""
import argparse
import json
import random
import hashlib
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chart_classifier import ROOT, CLASSES, build_model, load_image
from prepare_cv2 import read_labels


class Charts(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return load_image(ROOT / row['image']), CLASSES.index(row['label'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'results/cv2/manual_labels.csv')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-per-class', type=int, help='Essai limité par classe pour train et val uniquement')
    parser.add_argument('--cache-features', action='store_true', help='Calculer une seule fois le backbone gelé')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs et batch-size doivent être positifs')
    print('Validation des labels, chemins et empreintes des images...', flush=True)
    try:
        rows, image_hashes = read_labels(args.labels, return_hashes=True)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Entraînement bloqué : {error}\n')
    print(f'Validation réussie : {len(rows)} images.', flush=True)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model().to(device)
    selected = {}
    for split in ('train', 'val'):
        selected[split] = [r for r in rows if r['split'] == split]
        if args.max_per_class:
            selected[split] = [r for c in CLASSES for r in random.Random(42).sample(
                [x for x in selected[split] if x['label'] == c],
                min(args.max_per_class, sum(x['label'] == c for x in selected[split])))]
    # Le manifeste du checkpoint reflète exactement les images utilisées.
    rows = selected['train'] + selected['val'] + [r for r in rows if r['split'] == 'test']
    loaders = {}
    started = time.monotonic()
    for split in ('train', 'val'):
        dataset = Charts(selected[split])
        if args.cache_features:
            signature = {'rows': selected[split], 'hashes': [image_hashes[r['image']] for r in selected[split]],
                         'transform': 'RGB_resize224_imagenet_v1', 'backbone': 'resnet18_IMAGENET1K_V1'}
            cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
            cache = ROOT / 'results/cv2' / f'features_{split}_{cache_key}.pt'
            if cache.exists():
                saved = torch.load(cache, weights_only=True)
                features, labels = saved['features'], saved['labels']
            else:
                extractor = torch.nn.Sequential(*list(model.children())[:-1]).eval()
                chunks, targets = [], []
                with torch.inference_mode():
                    for index, (images, labels) in enumerate(DataLoader(dataset, batch_size=args.batch_size, num_workers=0)):
                        chunks.append(extractor(images.to(device)).flatten(1).cpu())
                        targets.append(labels)
                        if index % 50 == 0:
                            print(f'Features {split}: {min((index+1)*args.batch_size,len(dataset))}/{len(dataset)}', flush=True)
                features, labels = torch.cat(chunks), torch.cat(targets)
                torch.save({'features': features, 'labels': labels}, cache)
            dataset = TensorDataset(features, labels)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train', num_workers=0)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=.001)
    criterion = torch.nn.CrossEntropyLoss()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / 'results/cv2' / f'train_{stamp}'
    output.mkdir(parents=True)
    checkpoint = ROOT / 'models_cv2' / f'resnet18_{stamp}.pt'
    history, best = [], float('inf')
    for epoch in range(args.epochs):
        record = {'epoch': epoch + 1}
        for split, loader in loaders.items():
            # Backbone et BatchNorm gelés ; seule la tête est entraînée.
            model.eval()
            model.fc.train(split == 'train')
            total_loss, correct, total = 0., 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(split == 'train'):
                    logits = model.fc(images) if args.cache_features else model(images)
                    loss = criterion(logits, labels)
                    if split == 'train':
                        loss.backward()
                        optimizer.step()
                total_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
            record[f'{split}_loss'] = total_loss / total
            record[f'{split}_accuracy'] = correct / total
        history.append(record)
        print(record, flush=True)
        if record['val_loss'] < best:
            best = record['val_loss']
            checkpoint.parent.mkdir(exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'classes': CLASSES, 'trained': True,
                        'epoch': epoch + 1, 'rows': rows, 'image_hashes': {r['image']:image_hashes[r['image']] for r in rows},
                        'pretrained': 'IMAGENET1K_V1'}, checkpoint)
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    for metric in ('loss', 'accuracy'):
        fig, ax = plt.subplots()
        for split in ('train', 'val'):
            ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history], label=split)
        ax.set(xlabel='Epoch', ylabel=metric)
        ax.legend()
        fig.savefig(output / f'{metric}.png')
        plt.close(fig)
    summary = {'checkpoint': str(checkpoint), 'epochs': args.epochs, 'device': device,
               'best_validation_loss': best, 'seconds': time.monotonic() - started,
               'counts': {s:sum(r['split']==s for r in rows) for s in ('train','val','test')},
               'cache_features': args.cache_features, 'seed': 42}
    (output / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Checkpoint sélectionné sur validation : {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
