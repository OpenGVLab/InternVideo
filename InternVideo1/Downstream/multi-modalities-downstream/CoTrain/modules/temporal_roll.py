import torch
import torch.nn as nn
import random


class TemporalRoll(nn.Module):
    def __init__(self, n_segment=3, n_div=8, v=0):
        super(TemporalRoll, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        self.v = v

    def forward(self, x, layer=1):
        # return x
        nt, l, c = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, l, c)
        if self.v == 0:
            # 16, 3, 197, 768
            fold = l // self.fold_div
            out = torch.zeros_like(x)
            # keep cls token
            out[:, :, 0] = x[:, :, 0]
            #  roll left step 1 along time dimension (1)
            out[:, :, 1:fold+1] = torch.roll(x[:, :, 1:fold+1], 1, 1)
            # roll right step 1 along time dimension (1)
            out[:, :, -fold:] = torch.roll(x[:, :, -fold:], -1, 1)
            # not roll
            out[:, :, 1+fold:-fold] = x[:, :, 1+fold: -fold]
            # # 16, 3, 197, 768
            # fold = l // self.fold_div
            # out = torch.zeros_like(x)
            # #  roll left step 1 along time dimension (1)
            # out[:, :, :fold] = torch.roll(x[:, :, :fold], 1, 1)
            # # roll right step 1 along time dimension (1)
            # out[:, :, -fold:] = torch.roll(x[:, :, -fold:], -1, 1)
            # # not roll
            # out[:, :, fold:-fold] = x[:, :, fold: -fold]
        # random sampling
        elif self.v == 1:
            out = torch.zeros_like(x)
            roll_token_idexs = random.sample(range(1, l), l//2)
            # print(roll_token_idexs)
            out = x
            out[:, :, roll_token_idexs] = torch.roll(x[:, :, roll_token_idexs], 1, 1)
        # roll different tokens for different blocks
        elif self.v == 2:
            rolled_token_len = l // self.fold_div
            fold = rolled_token_len * (layer % self.fold_div)
            begin_index = 1 + fold
            end_index = min(1 + fold + rolled_token_len, l)
            out = torch.zeros_like(x)
            out[:, :, 0] = x[:, :, 0]  # cls token unchanged
            out[:, :, begin_index:] = x[:, :, begin_index:]
            out[:, :, begin_index:end_index] = torch.roll(x[:, :, begin_index:end_index], 1, 1)
            out[:, :, end_index:] = x[:, :, end_index:]
        else:  # not roll
            fold = c // self.fold_div
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left tokens
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right tokens
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, l, c)