"""Extraire seulement les JSON d'annotations du ZIP officiel via HTTP Range."""
import argparse
import io
import json
import struct
import zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://huggingface.co/datasets/ahmed-masry/ChartQA'


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position, self.blocks = url, 0, []
        response = requests.get(url, headers={'Range': 'bytes=-65536'}, timeout=90)
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError('Serveur sans HTTP Range : téléchargement complet refusé')
        self.size = int(response.headers['Content-Range'].split('/')[-1])
        self.blocks.append((self.size - len(response.content), response.content))
        self.transferred = len(response.content)

    def fetch(self, start, end):
        for offset, block in self.blocks:
            if offset <= start and end <= offset + len(block):
                return block[start-offset:end-offset]
        response = requests.get(self.url, headers={'Range': f'bytes={start}-{end-1}'}, timeout=120)
        response.raise_for_status()
        expected = f'bytes {start}-{end-1}/{self.size}'
        if response.status_code != 206 or response.headers.get('Content-Range') != expected:
            raise RuntimeError(f'Réponse Range invalide : {response.status_code}, {response.headers.get("Content-Range")}')
        self.blocks.append((start, response.content))
        self.transferred += len(response.content)
        return response.content

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position + size, self.size)
        data = self.fetch(self.position, end)
        self.position = end
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    metadata = requests.get('https://huggingface.co/api/datasets/ahmed-masry/ChartQA', timeout=60)
    metadata.raise_for_status()
    revision = metadata.json()['sha']
    url = f'{REPO}/resolve/{revision}/ChartQA%20Dataset.zip'
    remote = RemoteZip(url)
    with zipfile.ZipFile(remote) as archive:
        infos = sorted([i for i in archive.infolist() if '/annotations/' in i.filename and i.filename.endswith('.json')], key=lambda i:i.header_offset)
        if not infos:
            raise RuntimeError('Aucune annotation dans le ZIP officiel')
        print('Revision', revision, 'annotations', len(infos), 'compressed bytes', sum(i.compress_size for i in infos), flush=True)
        if args.inspect:
            print([i.filename for i in infos[:3]])
            return
        # Ranges contigus : n'inclure que des entrées annotation, jamais les PNG.
        groups = []
        all_infos = sorted(archive.infolist(), key=lambda i:i.header_offset)
        selected = set(i.filename for i in infos)
        for info in all_infos:
            if info.filename in selected:
                if not groups or previous not in selected:
                    groups.append([])
                groups[-1].append(info)
            previous = info.filename
        destination = ROOT / 'annotations_cv2' / revision
        destination.mkdir(parents=True, exist_ok=True)
        for group in groups:
            last = group[-1]
            header = remote.fetch(last.header_offset, last.header_offset + 30)
            name_length, extra_length = struct.unpack_from('<HH', header, 26)
            end = last.header_offset + 30 + name_length + extra_length + last.compress_size
            remote.fetch(group[0].header_offset, end)
            print('Annotation range:', len(group), 'files', flush=True)
            for info in group:
                parts = Path(info.filename).parts
                split = parts[parts.index('annotations') - 1]
                if split not in ('train', 'val', 'test'):
                    raise ValueError('Split inconnu')
                target = destination / split / Path(info.filename).name
                target.parent.mkdir(exist_ok=True)
                data = archive.read(info)  # CRC contrôlé par zipfile.
                json.loads(data)
                if not target.exists():
                    target.write_bytes(data)
                elif target.read_bytes() != data:
                    raise RuntimeError(f'Fichier existant différent : {target}')
        report = {'repository': REPO, 'revision': revision, 'archive_url': url,
                  'archive_size': remote.size, 'transferred_bytes': remote.transferred,
                  'annotations': len(infos), 'images_downloaded': 0}
        (destination / 'provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
