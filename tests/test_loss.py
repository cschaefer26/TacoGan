import unittest
import torch

from losses import MaskedL1


class TestLoss(unittest.TestCase):

    def test_masked_l1(self):
        criterion = MaskedL1()
        inp = torch.tensor([[1, 2], [1, 2]]).unsqueeze(-1)
        tar = torch.tensor([[2, 3], [3, 4]]).unsqueeze(-1)
        lens = torch.tensor([1, 2])
        loss = criterion(inp, tar, lens)
        expected = (1 + 2 + 2) / 3.
        self.assertAlmostEqual(expected, float(loss))

        lens = torch.tensor([2, 2])
        loss = criterion(inp, tar, lens)
        expected = (1 + 1 + 2 + 2) / 4.
        self.assertAlmostEqual(expected, float(loss))



