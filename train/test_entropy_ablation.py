import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from utils_update import CrossBatchMemoryV2, sample_triplets_v13


class EntropyAblationTests(unittest.TestCase):
    def test_only_gate_changes_partners_and_default(self):
        bank = CrossBatchMemoryV2(2, 8, 3, device='cpu')
        bank.update(torch.tensor([[1., 0.], [0., 1.], [-1., 0.]]),
                    torch.tensor([[0., 1., 0.], [0., 1., 0.], [0., 0., 1.]]))
        embeddings = torch.tensor([[.8, .2], [.2, .8]], requires_grad=True)
        labels = torch.tensor([[0., 1., 0.], [0., 1., 0.]])
        logits = torch.tensor([[0., 0., 0.], [10., 0., 0.]])
        stats = {}
        all_trips = sample_triplets_v13(embeddings, logits, labels, bank, head_class_ids=[0],
                                       anchor_rule='all', diagnostics=stats)
        self.assertEqual(stats['valid'], 2)
        rng = torch.get_rng_state().clone()
        gated = sample_triplets_v13(embeddings, logits, labels, bank, head_class_ids=[0])
        explicit = sample_triplets_v13(embeddings, logits, labels, bank, head_class_ids=[0], anchor_rule='mean')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for all_part, gated_part, explicit_part in zip(all_trips, gated, explicit):
            self.assertTrue(torch.equal(all_part[:1], gated_part))
            self.assertTrue(torch.equal(gated_part, explicit_part))
        self.assertFalse(gated[1].requires_grad)
        self.assertFalse(gated[2].requires_grad)

    def test_empty_memory_and_equal_entropy(self):
        bank = CrossBatchMemoryV2(2, 4, 2, device='cpu')
        x = torch.tensor([[1., 0.], [0., 1.]])
        labels = torch.eye(2)
        self.assertIsNone(sample_triplets_v13(x, torch.zeros(2, 2), labels, bank))
        bank.update(x, labels)
        a = sample_triplets_v13(x, torch.zeros(2, 2), labels, bank, anchor_rule='all')
        b = sample_triplets_v13(x, torch.zeros(2, 2), labels, bank, anchor_rule='mean')
        for first, second in zip(a, b):
            self.assertTrue(torch.equal(first, second))

    def test_three_arm_cpu_training_fixed_endpoint_and_group_ap(self):
        import train_2_v3 as v3

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 3)

            def forward(self, x):
                logits = self.head(x.mean((2, 3)))
                self.head.last_h = logits.unsqueeze(1)
                return logits, logits.unsqueeze(1)

        torch.manual_seed(1)
        model = Tiny()
        batches = [(torch.rand(4, 3, 4, 4), torch.tensor([[1., 0., 0.], [0., 1., 0.],
                                                          [0., 0., 1.], [0., 1., 0.]])) for _ in range(2)]
        initial = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.cuda, 'is_available', return_value=False):
            for arm, weight, gate in [('baseline', 0., 'all'), ('triplet', .1, 'all'), ('entropy', .1, 'mean')]:
                cfg = dict(num_classes=3, lr=.001, weight_decay=.01, embedding_dim=3, memory_size=8,
                           gamma_neg=2, gamma_pos=0, margin=.5, epochs=2, patience=0, monitor='mAP',
                           train_size=8, batch_size=4, accum_steps=1, use_amp=False, log_every=100,
                           fixed_budget=True, triplet_lambda=weight, anchor_rule=gate,
                           ap_groups=dict(head=[0], medium=[1], tail=[2]), out_dir=str(Path(tmp) / arm))
                with patch.object(v3, 'create_model', return_value=copy.deepcopy(model)):
                    trainer = v3.Trainer(cfg, [1., 1., 1.])
                trainer.fit(batches, batches)
                history = json.loads((Path(tmp) / arm / 'history.json').read_text())
                self.assertEqual(len(history['epoch_metrics']), 2)
                self.assertTrue((Path(tmp) / arm / 'model_final.pth').exists())
                final = history['epoch_metrics'][-1]
                for c, group in enumerate(['head', 'medium', 'tail']):
                    self.assertEqual(final[group + '_AP'], final['per_class_AP'][c])
                initial.append(history['initial_metrics'])
                if arm == 'baseline':
                    self.assertEqual(trainer.memory.filled, 0)
                else:
                    self.assertEqual(trainer.memory.filled, 8)
                    self.assertGreater(history['triplet_diagnostics'][-1]['valid'], 0)
            self.assertEqual(initial[0], initial[1])
            self.assertEqual(initial[1], initial[2])


if __name__ == '__main__':
    unittest.main()
