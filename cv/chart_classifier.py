"""ResNet18 couleur pour CV2. Une tête non entraînée ne constitue pas un classificateur."""
from pathlib import Path
import argparse
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['v_bar', 'h_bar', 'line', 'pie']
# Cache limité au nouveau dossier CV2, sans écriture dans les modèles existants.
torch.hub.set_dir(str(ROOT / 'models_cv2' / 'torch_cache'))


def image_transform():
    # Pas de crop : garder les axes et légendes. Resize déforme le ratio d'origine.
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def load_image(path):
    with Image.open(path) as image:
        return image_transform()(image.convert('RGB'))


def build_model(classes=CLASSES, pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    return model


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not checkpoint.get('trained', False):
        raise ValueError('Checkpoint technique non entraîné : prédiction métier interdite.')
    model = build_model(checkpoint['classes'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint)
    torch.set_num_threads(4)
    predictions = []
    with torch.inference_mode():
        for path in args.image:
            probabilities = model(load_image(path).unsqueeze(0)).softmax(1)[0]
            predictions.append({'image': str(path), 'prediction': checkpoint['classes'][probabilities.argmax().item()],
                                'probabilities': dict(zip(checkpoint['classes'], probabilities.tolist()))})
    print(json.dumps(predictions, indent=2))


if __name__ == '__main__':
    main()
