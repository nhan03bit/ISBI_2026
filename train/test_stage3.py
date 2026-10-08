"""Run: python -m unittest discover -s train -p test_stage3.py"""
import csv
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stage3_data import build_index, read_bytes, schedule, exposure


class SamplingTests(unittest.TestCase):
    def test_patient_cycle(self):
        records = [dict(patient=str(i // 2), labels=[int(i < 8)]) for i in range(1000)]
        index = dict(records=records, names=['rare'], counts=[8], total=1000)
        draws = schedule(index, 24, targeted_fraction=.25)
        patients = [records[i]['patient'] for b in draws for i, _, c in b if c == 0]
        for offset in range(0, 24, 4):
            self.assertEqual(len(set(patients[offset:offset + 4])), 4)

    def test_targeting_and_resume_schedule(self):
        records = [dict(patient=str(i // 2), labels=[int(i < 2), int(i < 4), 1]) for i in range(400)]
        index = dict(records=records, names=['rare', 'boundary', 'common'], counts=[2, 4, 400], total=400)
        draws = schedule(index, 100, seed=5)
        self.assertEqual(draws, schedule(index, 100, seed=5))
        self.assertEqual(exposure(index, draws)['targeted_exposure'], {0: 50})
        self.assertEqual(len(draws[40:]), 60)
        occurrences = {}
        for batch in draws:
            self.assertEqual(len({i for i, _, _ in batch}), 4)
            for i, occurrence, c in batch:
                self.assertEqual(occurrence, occurrences.get(i, 0))
                occurrences[i] = occurrence + 1
                if c >= 0:
                    self.assertEqual(records[i]['labels'][c], 1)
        natural = schedule(index, 100, targeted_fraction=0)
        self.assertEqual(len({i for b in natural for i, _, _ in b}), 400)

    def test_index_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'fold.csv').write_text('filename,label\na.png,"[\'a\']"\nb.png,"[]"\n')
            (root / 'meta.csv').write_text('ImageID,PatientID\na.png,001\nb.png,002\n')
            with tarfile.open(root / 'shard_train_0000.tar', 'w') as tar:
                for ext, data in [('img', b'image'), ('cls', b'label')]:
                    item = tarfile.TarInfo('train_00000000.' + ext)
                    item.size = len(data)
                    tar.addfile(item, io.BytesIO(data))
            idx = build_index(root, root / 'fold.csv', root / 'meta.csv', ['a'])
            self.assertEqual(idx['missing'], ['b.png'])
            self.assertEqual(idx['counts'], [1])
            self.assertEqual(read_bytes(idx['records'][0]['img']), b'image')


class TrainingTests(unittest.TestCase):
    def test_paired_bootstrap(self):
        import importlib.util
        import numpy as np
        path = Path(__file__).resolve().parents[1] / 'analysis/stage3_compare.py'
        spec = importlib.util.spec_from_file_location('compare', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        data = dict(keys=np.arange(4), labels=y, scores=y * .8 + .1, names=np.array(['a', 'b']),
                    patients=np.array(['p1', 'p1', 'p2', 'p2']), unseen=np.ones(4, dtype=bool),
                    tail=np.array([1]), head=np.array([0]))
        report = module.compare(data, data, repeats=10)
        self.assertEqual(report['groups']['tail']['interval95'], [0., 0.])
        self.assertEqual(report['groups']['overall']['valid_bootstraps'], 10)
        bad = dict(data, keys=np.arange(4)[::-1])
        with self.assertRaises(ValueError):
            module.compare(data, bad, repeats=1)

    def test_cpu_training_export_and_exact_resume(self):
        import torch
        from PIL import Image
        import train_3
        import train_2_v3 as v3

        class Tiny(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 3)
                self.head = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(3, 30))

            def forward(self, x):
                x = self.backbone(x.mean((2, 3)))
                return self.head(x), x

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [f'c{i}' for i in range(30)]
            torch.manual_seed(5)
            initial = Tiny().state_dict()
            torch.save(initial, root / 'init.pt')
            metadata = []
            for split in ('train', 'val'):
                directory = root / split
                directory.mkdir()
                torch.save({'class_names': names}, directory / 'label_info.pt')
                rows = []
                with tarfile.open(directory / f'shard_{split}_0000.tar', 'w') as tar:
                    for i in range(4):
                        filename = f'{split}{i}.png'
                        metadata.append([filename, f'{split}{i}'])
                        labels = [int((i + c) % 2 == 0) for c in range(30)]
                        rows.append([filename, str([names[c] for c in range(30) if labels[c]])])
                        image = io.BytesIO()
                        Image.new('RGB', (20, 20), (50 + i * 20, 80, 100)).save(image, format='PNG')
                        target = io.BytesIO()
                        torch.save(torch.tensor(labels, dtype=torch.float32), target)
                        for ext, data in [('img', image.getvalue()), ('cls', target.getvalue())]:
                            member = tarfile.TarInfo(f'{split}_{i:08d}.{ext}')
                            member.size = len(data)
                            tar.addfile(member, io.BytesIO(data))
                with open(root / f'{split}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['filename', 'label'])
                    writer.writerows(rows)
            with open(root / 'metadata.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ImageID', 'PatientID'])
                writer.writerows(metadata)
            base = ['train_3.py', '--checkpoint', str(root / 'init.pt'), '--train-shards', str(root / 'train'),
                    '--val-shards', str(root / 'val'), '--train-fold', str(root / 'train.csv'),
                    '--val-fold', str(root / 'val.csv'), '--metadata', str(root / 'metadata.csv'),
                    '--arm', 'natural', '--updates', '4', '--accum-steps', '2', '--img-size', '16']
            with patch.object(v3, 'ConvNeXt2', Tiny):
                for output, extra in [('full', []), ('resumed', ['--stop-after', '1']), ('resumed', ['--resume'])]:
                    with patch.object(sys, 'argv', base + ['--out', str(root / output)] + extra):
                        train_3.main()
            full = torch.load(root / 'full/model_000004.pth', weights_only=True)
            resumed = torch.load(root / 'resumed/model_000004.pth', weights_only=True)
            for k in full:
                self.assertTrue(torch.equal(full[k], resumed[k]), k)
                if k.startswith('backbone.'):
                    self.assertTrue(torch.equal(full[k], initial[k]), k)
            self.assertFalse(torch.equal(full['head.1.weight'], initial['head.1.weight']))
            self.assertEqual(len(json.loads((root / 'full/metrics.json').read_text())), 3)


if __name__ == '__main__':
    unittest.main()
