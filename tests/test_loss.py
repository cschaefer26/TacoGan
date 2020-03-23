import unittest
import torch
from losses import MaskedL1, MaskedBCE


class TestLoss(unittest.TestCase):

    def test_masked_l1(self):
        masked_l1 = MaskedL1()
        x = torch.tensor([[1, 2], [1, 2]]).unsqueeze(-1)
        tar = torch.tensor([[2, 3], [3, 4]]).unsqueeze(-1)
        lens = torch.tensor([1, 2])
        loss = masked_l1(x, tar, lens)
        expected = (1 + 2 + 2) / 3.
        self.assertAlmostEqual(expected, float(loss))

        lens = torch.tensor([2, 2])
        loss = masked_l1(x, tar, lens)
        expected = (1 + 1 + 2 + 2) / 4.
        self.assertAlmostEqual(expected, float(loss))

    def test_masked_ce(self):
        masked_ce = MaskedBCE(pos_weight=10)
        x = torch.tensor([[0, 0, 1], [0, 0, 1]])
        tar = torch.tensor([[0, 0, 1], [0, 1, 0]])
        lens = torch.tensor([3, 2])
        loss = masked_ce(x, tar, lens)
        self.assertAlmostEqual(2.428705930709839, float(loss))

        x = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]).float()
        tar = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]]).float()
        lens = torch.tensor([3, 2])
        loss = masked_ce(x, tar, lens)
        self.assertAlmostEqual(2.428705930709839, float(loss))

        print(loss)


