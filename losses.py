import torch.nn.functional as F
import torch


# Adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def pad_mask(lens, max_len):
    batch_size = lens.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range = seq_range.unsqueeze(0)
    seq_range = seq_range.expand(batch_size, max_len)
    if lens.is_cuda:
        seq_range = seq_range.cuda()
    lens = lens.unsqueeze(1)
    lens = lens.expand_as(seq_range)
    mask = seq_range < lens
    return mask.unsqueeze(2).float()


class MaskedL1(torch.nn.Module):

    def __call__(self, x, target, lens):
        target.requires_grad = False
        max_len = target.size(1)
        mask = pad_mask(lens, max_len)
        mask = mask.expand_as(x)
        loss = F.l1_loss(
            x * mask, target * mask, reduction="sum")
        return loss / mask.sum()