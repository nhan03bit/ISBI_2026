"""Indexed uncompressed WebDataset access and reproducible Phase A schedules."""
import ast
import csv
import hashlib
import json
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_index(shards, fold, metadata, names):
    with open(fold, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    with open(metadata, newline='', encoding='utf-8-sig') as f:
        patients = {}
        for row in csv.DictReader(f):
            key = row['ImageID']
            if key in patients or not row['PatientID'].strip():
                raise ValueError('Duplicate image or missing patient in metadata')
            patients[key] = row['PatientID']
    if len({r['filename'] for r in rows}) != len(rows):
        raise ValueError('Duplicate fold filenames')
    row_labels = [set(ast.literal_eval(r['label'])) for r in rows]
    if any(labels - set(names) for labels in row_labels):
        raise ValueError('Fold contains unknown labels')
    by_name = {r['filename']: i for i, r in enumerate(rows)}
    members = defaultdict(dict)
    inventory = []
    for path in sorted(Path(shards).glob('shard_*.tar')):
        stat = path.stat()
        inventory.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                key, ext = member.name.rsplit('.', 1)
                if ext not in ('img', 'cls'):
                    continue
                if not member.isfile() or ext in members[key]:
                    raise ValueError('Invalid or duplicate tar member')
                members[key][ext] = [str(path.resolve()), member.offset_data, member.size]
    if not inventory:
        raise ValueError('No uncompressed shard_*.tar files found')
    records, seen = [], set()
    for key, parts in sorted(members.items()):
        if set(parts) != {'img', 'cls'}:
            raise ValueError(f'Incomplete shard record: {key}')
        if key in by_name:
            i = by_name[key]
        else:
            prefix, number = key.rsplit('_', 1)
            if prefix not in ('train', 'val') or not number.isdigit():
                raise ValueError(f'Unknown key format: {key}')
            i = int(number)
        if i >= len(rows) or i in seen:
            raise ValueError('Invalid or duplicate fold row in shards')
        seen.add(i)
        row = rows[i]
        labels = row_labels[i]
        patient = patients[row['filename']]
        records.append(dict(key=key, filename=row['filename'],
                            patient=hashlib.sha256(patient.encode()).hexdigest(),
                            labels=[int(c in labels) for c in names], **parts))
    # Membership denominator includes missing images; never silently redefine rarity.
    counts = [sum(c in labels for labels in row_labels) for c in names]
    return dict(records=records, names=list(names), counts=counts, total=len(rows),
                missing=[r['filename'] for i, r in enumerate(rows) if i not in seen],
                inventory=inventory, fold_hash=sha256(fold), metadata_hash=sha256(metadata))


def read_bytes(location):
    path, offset, length = location
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read(length)
    if len(data) != length:
        raise ValueError('Truncated shard')
    return data


def schedule(index, batches, batch_size=4, targeted_fraction=0.125, seed=42):
    """Immutable draw list; (record index, occurrence, target label or -1)."""
    if batches < 1 or batch_size < 1 or not 0 <= targeted_fraction <= 1 / batch_size:
        raise ValueError('At most one targeted draw per batch is supported')
    records = index['records']
    if len(records) < batch_size:
        raise ValueError('Too few distinct images for batch size')
    tail = [c for c, n in enumerate(index['counts']) if 0 < n / index['total'] < .01]
    rng = random.Random(seed)
    pools = {c: defaultdict(list) for c in tail}
    for i, r in enumerate(records):
        for c in tail:
            if r['labels'][c]:
                pools[c][r['patient']].append(i)
    if targeted_fraction and (not tail or any(not pools[c] for c in tail)):
        raise ValueError('Minority label has no available positive patients')
    patient_cycles, image_cycles, natural = {}, {}, []
    occurrences = Counter()
    output = []
    for b in range(batches):
        selected = []
        # Deterministic evenly spaced targeting, exact for half the batches.
        rate = targeted_fraction * batch_size
        if int((b + 1) * rate) > int(b * rate):
            c = rng.choices(tail, weights=[len(pools[c]) ** .5 for c in tail])[0]
            if not patient_cycles.get(c):
                patient_cycles[c] = list(pools[c])
                rng.shuffle(patient_cycles[c])
            p = patient_cycles[c].pop()
            if not image_cycles.get((c, p)):
                image_cycles[c, p] = list(pools[c][p])
                rng.shuffle(image_cycles[c, p])
            selected.append((image_cycles[c, p].pop(), c))
        while len(selected) < batch_size:
            if not natural:
                natural = list(range(len(records)))
                rng.shuffle(natural)
            i = natural.pop()
            if all(i != existing for existing, _ in selected):
                selected.append((i, -1))
        batch = []
        for i, target in selected:
            batch.append((i, occurrences[i], target))
            occurrences[i] += 1
        output.append(batch)
    return output


def exposure(index, draws):
    counts = Counter(i for batch in draws for i, _, _ in batch)
    patients = Counter()
    for i, n in counts.items():
        patients[index['records'][i]['patient']] += n
    return dict(images=sum(counts.values()), unique_images=len(counts),
                unique_patients=len(patients), image_repeat_histogram=dict(Counter(counts.values())),
                patient_repeat_histogram=dict(Counter(patients.values())),
                positive_exposure=[sum(n * index['records'][i]['labels'][c] for i, n in counts.items())
                                   for c in range(len(index['names']))],
                positive_patients=[len({index['records'][i]['patient'] for i in counts
                                        if index['records'][i]['labels'][c]}) for c in range(len(index['names']))],
                targeted_exposure=dict(Counter(c for b in draws for _, _, c in b if c >= 0)))


class IndexedImages:
    def __init__(self, index, train, size, seed):
        self.index, self.train, self.size, self.seed = index, train, size, seed

    def __len__(self):
        return len(self.index['records'])

    def __getitem__(self, draw):
        import io
        import torch
        import train_2_v3 as v3
        i, occurrence, _ = draw if isinstance(draw, tuple) else (draw, 0, -1)
        r = self.index['records'][i]
        label_bytes = read_bytes(r['cls'])
        labels = torch.load(io.BytesIO(label_bytes), weights_only=True)
        expected = torch.tensor(r['labels'], dtype=torch.float32)
        if labels.shape != expected.shape or not torch.equal(labels, expected):
            raise ValueError(f'Shard/fold label mismatch: {r["key"]}')
        v3.AUG_BASE_SEED = self.seed
        result = v3.decode_and_transform(dict(__key__=r['key'], img=read_bytes(r['img']), cls=label_bytes),
                                         is_train=self.train, epoch=occurrence, size=self.size)
        if result is None:
            raise ValueError(f'Cannot decode {r["key"]}; refusing silent exclusion')
        return result
