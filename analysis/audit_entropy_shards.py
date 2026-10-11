"""Record unreadable image keys before submitting matched entropy ablations."""
import argparse
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np


def inventory(directory):
    paths = sorted(Path(directory).glob('shard_*.tar'))
    if not paths:
        raise ValueError(f'No shards in {directory}')
    return [[str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns] for path in paths]


def audit(train_dir, val_dir):
    output = {'inventory': {'train': inventory(train_dir), 'val': inventory(val_dir)},
              'images': {}, 'bad_keys': {}}
    for split in ('train', 'val'):
        bad, keys, count = [], set(), 0
        for path, _, _ in output['inventory'][split]:
            with tarfile.open(path, 'r:') as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith('.img'):
                        continue
                    key = member.name[:-4]
                    if key in keys:
                        raise ValueError(f'Duplicate {split} key: {key}')
                    keys.add(key)
                    count += 1
                    stream = archive.extractfile(member)
                    data = stream.read() if stream else b''
                    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
                        bad.append(key)
        output['images'][split] = count
        output['bad_keys'][split] = bad
        print(f'{split}: {count} images; {len(bad)} unreadable: {bad}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = audit(args.train, args.val)
    Path(args.out).write_text(json.dumps(result, indent=2))
