"""Frozen feature extraction must preserve BatchNorm state and backbone weights."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from flower_training import run_epoch, state_fingerprint
from flower_transfer import FlowerResNet18, backbone_fingerprint


class TransferLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def batch(self):
        torch.manual_seed(42)
        return DataLoader(TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])), batch_size=2)

    def test_head_training_changes_only_the_head_and_keeps_batchnorm_frozen(self):
        model = FlowerResNet18(stage="head")
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
        run_epoch(model, self.batch(), torch.device("cpu"), optimizer, log_every=0)
        self.assertEqual(backbone_fingerprint(before), backbone_fingerprint(model.state_dict()))
        self.assertNotEqual(state_fingerprint(before), state_fingerprint(model.state_dict()))
        self.assertTrue(all(not module.training for module in model.network.modules() if isinstance(module, torch.nn.BatchNorm2d)))
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith("network.fc.")))

    def test_fine_tuning_unfreezes_backbone_and_updates_batchnorm(self):
        model = FlowerResNet18(stage="head")
        model.set_stage("finetune")
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        first_conv = model.network.conv1.weight.detach().clone()
        tracked = model.network.bn1.num_batches_tracked.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        run_epoch(model, self.batch(), torch.device("cpu"), optimizer, log_every=0)
        self.assertFalse(torch.equal(first_conv, model.network.conv1.weight))
        self.assertEqual(model.network.bn1.num_batches_tracked.item(), tracked.item() + 1)
        self.assertTrue(model.network.bn1.training)


if __name__ == "__main__":
    unittest.main()
