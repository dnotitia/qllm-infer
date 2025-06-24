import numpy as np
import math
import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch import nn
from torch.autograd import Function, Variable

class QRazor(Function):
    @staticmethod
    def forward(ctx, x, sign, q_bit, r_bit, group):
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
            
            if outlier_id.any():
                rounded_values = torch.round(raw_x[outlier_id] / round_value)
                outlier_signs = sign.view(-1)[:len(raw_x.view(-1))].view(-1, group)[outlier_id]
                condition = (rounded_values == (2**(r_bit-1))) & (outlier_signs > 0)
                rounded_values = torch.where(condition, torch.tensor(2**(r_bit-1)-1, device=rounded_values.device), rounded_values)
                raw_x[outlier_id] = rounded_values * round_value
                
        raw_x = raw_x.view(-1)
        x = raw_x[:org_len].view_as(x)
        return x
