import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from audit_entropy_shards import audit


class AuditTests(unittest.TestCase):
    def test_unreadable_key_is_recorded_for_both_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val'):
                directory = root / split
                directory.mkdir()
                ok, encoded = cv2.imencode('.png', np.full((8, 8, 3), 100, np.uint8))
                self.assertTrue(ok)
                with tarfile.open(directory / 'shard_0000.tar', 'w') as archive:
                    for key, data in [('good', encoded.tobytes()), ('bad', b'not an image')]:
                        member = tarfile.TarInfo(f'{key}.img')
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))
            result = audit(root / 'train', root / 'val')
            self.assertEqual(result['bad_keys'], dict(train=['bad'], val=['bad']))
            self.assertEqual(result['images'], dict(train=2, val=2))


if __name__ == '__main__':
    unittest.main()
