import numpy as np
import math
import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch import nn
from torch.autograd import Function, Variable

class QRazor(Function):
    @staticmethod
    def forward(ctx, x, q_bit, r_bit, group):
        raw_x = torch.reshape(x, (-1,))
        org_len = len(raw_x)
        if org_len % group:
            vacant_num = group - org_len % group
            raw_x = F.pad(raw_x, (0, vacant_num), 'constant', 0)
        raw_x = raw_x.view(-1, group)
        max_dim1, _ = raw_x.max(dim=1)

        for b in range(r_bit, q_bit+1):
            mul_xth = 2 ** (b - 1)
            round_value = 2 ** (b + 1 - r_bit)
            outlier_id = (max_dim1 >= mul_xth) & (max_dim1 < mul_xth * 2)
            cond2 = max_dim1 >= (2 * mul_xth - 2 ** (b - 4))

            if outlier_id.any():
                threshold = math.floor(2 * mul_xth - 2 ** (b - 4))
                indices_both = outlier_id & cond2
                indices_round = outlier_id & (~cond2)

                if indices_both.any():
                    selected_raw_x = raw_x[indices_both]
                    element_floor = selected_raw_x >= threshold
                    element_round = selected_raw_x < threshold
                    selected_raw_x[element_floor] = torch.floor(selected_raw_x[element_floor] / round_value) * round_value
                    selected_raw_x[element_round] = torch.round(selected_raw_x[element_round] / round_value) * round_value
                    raw_x[indices_both] = selected_raw_x

                if indices_round.any():
                    raw_x[indices_round] = torch.round(raw_x[indices_round] / round_value) * round_value
        raw_x = raw_x.view(-1)
        x = raw_x[:org_len].view_as(x)

        return x