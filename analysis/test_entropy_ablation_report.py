import json
import tempfile
import unittest
from pathlib import Path
from entropy_ablation_report import ARMS, report


class ReportTests(unittest.TestCase):
    def test_matched_results_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm, (weight, gate) in ARMS.items():
                directory = root / 'seed_42' / arm
                directory.mkdir(parents=True)
                cfg = dict(seed=42, epochs=4, fixed_budget=True, strict_init=True, swa_last_k=0,
                           limit_train_batches=0, limit_val_batches=0, img_size=768,
                           out_dir=str(directory), triplet_lambda=weight, anchor_rule=gate)
                (directory / 'manifest.json').write_text(json.dumps(dict(config=cfg, checkpoint_sha256='same')))
                (directory / 'history.json').write_text(json.dumps(dict(initial_metrics={'mAP': .3},
                    triplet_diagnostics=[dict(anchors=8)] * 4,
                    epoch_metrics=[dict(epoch=4, mAP=.4, head_AP=.6, medium_AP=.4, tail_AP=.2)])))
                (directory / 'model_final.pth').touch()
            rows, table = report(root)
            self.assertEqual(len(rows), 3)
            self.assertIn('entropy minus triplet', table)
            path = root / 'seed_42/entropy/manifest.json'
            bad = json.loads(path.read_text())
            bad['config']['img_size'] = 512
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'Unmatched'):
                report(root)


if __name__ == '__main__':
    unittest.main()
