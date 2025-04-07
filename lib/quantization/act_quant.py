from types import MethodType
import logging
import torch.nn as nn
import torch.nn.functional as F

from . import attention
from lib.quantization.quantizer import *
from lib.quantization.metric import *

class ActQuantLinear(nn.Linear):
    def __init__(self,in_features,out_features,bias,args):
        super().__init__(in_features,out_features,bias)
        self.perchannel = not args.a_per_tensor
        self.pertoken_a = args.a_per_token
        self.groupsize = args.groupsize_a
        self.bit = args.bits_a
        self.sym = args.sym_a
        self.quantizer = Quantizer()
        self.quantizer.configure(
            self.bit, self.perchannel, sym=self.sym, mse=False
        )
        if args.analyze_stats:
            self.stats = {
                'stat_sqnr_x': [],
                'stat_sqnr_w': [],
                'stat_sqnr_o': [],
                'stat_mse_x': [],
                'stat_mse_w': [],
                'stat_mse_o': [],
                'stat_mean_x': [],
                'stat_mean_w': [],
                'stat_std_x': [],
                'stat_std_w': [],
                'stat_kurt_x': [],
                'stat_kurt_w': [],
                'stat_max_x': [],
                'stat_max_w': [],
                'stat_min_x': [],
                'stat_min_w': [],
            }
        else:
            self.stats = None

    def forward(self, x):
        if self.bit < 16:
            shape_ = x.shape
            if self.groupsize > 0:
                qx = x.reshape(-1, self.groupsize)
            else: # Token-wise
                qx = x.reshape(-1, shape_[-1])
            self.quantizer.find_params(qx, weight=self.pertoken_a)
            qx = quantize(
                qx, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq, self.sym
            ).to(self.weight.dtype)
            qx = qx.reshape(shape_)
        else:
            qx = x

        out = F.linear(qx,self.weight,self.bias)

        if self.stats is not None:
            with torch.no_grad():
                ref_output = F.linear(x,self.fp_weight.to(x.device))
                self.stats['stat_sqnr_x'].append(sqnr(x.data, qx))
                self.stats['stat_sqnr_w'].append(sqnr(self.fp_weight.data, self.weight.data))
                self.stats['stat_sqnr_o'].append(sqnr(ref_output, out))
                self.stats['stat_mse_x'].append(mse(x.data, qx))
                self.stats['stat_mse_w'].append(mse(self.fp_weight.data, self.weight.data))
                self.stats['stat_mse_o'].append(mse(ref_output, out))
                self.stats['stat_mean_x'].append(x.mean().item())
                self.stats['stat_mean_w'].append(self.fp_weight.mean().item())
                self.stats['stat_std_x'].append(x.std().item())
                self.stats['stat_std_w'].append(self.fp_weight.std().item())
                self.stats['stat_kurt_x'].append(kurtosis(x))
                self.stats['stat_kurt_w'].append(kurtosis(self.fp_weight))
                self.stats['stat_max_x'].append(x.max().item())
                self.stats['stat_max_w'].append(self.fp_weight.max().item())
                self.stats['stat_min_x'].append(x.min().item())
                self.stats['stat_min_w'].append(self.fp_weight.min().item())
            for k, v in self.stats.items():
                self.register_buffer(k,torch.tensor(v))

        return out

class ActQuantMatMul1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.perchannel_q = not args.q_per_tensor
        self.pertoken_q = args.q_per_token
        self.groupsize_q = args.groupsize_q
        assert self.groupsize_q < 0, 'Group-wise quantization on multi-head activation is not supported yet'
        self.bit_q = args.bits_q
        self.sym_q = args.sym_q
        self.perchannel_k = not args.k_per_tensor
        self.pertoken_k = args.k_per_token
        self.groupsize_k = args.groupsize_k
        assert self.groupsize_k < 0, 'Group-wise quantization on multi-head activation is not supported yet'
        self.bit_k = args.bits_k
        self.sym_k = args.sym_k
        self.quantizer_A = Quantizer()
        self.quantizer_A.configure(
            self.bit_q, self.perchannel_q, sym=self.sym_q, mse=False
        )
        self.quantizer_B = Quantizer()
        self.quantizer_B.configure(
            self.bit_k, self.perchannel_k, sym=self.sym_k, mse=False
        )

    def forward(self, A, B): # [b, h, s, d]
        if (self.bit_q or self.bit_k) < 16:
            if (self.bit_k) >= 16: 
                # Quantizing A
                Ashape_ = A.shape
                A = A.reshape(-1, Ashape_[-1])
                self.quantizer_A.find_params(A, weight=self.pertoken_q)
                qA = quantize(
                    A, self.quantizer_A.scale, self.quantizer_A.zero, self.quantizer_A.maxq, self.sym_q
                ).to(A.dtype)
                qA = qA.reshape(Ashape_)
                # Not Quantizing B
                qB = B
            elif (self.bit_q) >= 16:
                # Not Quantizing A 
                qA = A
                # Quantizing B
                B = B.transpose(-1,-2)
                Bshape_ = B.shape
                B = B.reshape(-1, Bshape_[-1])
                self.quantizer_B.find_params(B, weight=self.pertoken_k)
                qB = quantize(
                    B, self.quantizer_B.scale, self.quantizer_B.zero, self.quantizer_B.maxq, self.sym_k
                ).to(B.dtype)
                qB = qB.reshape(Bshape_)
                qB = qB.transpose(-1,-2)
            else:
                # Quantizing A
                Ashape_ = A.shape
                A = A.reshape(-1, Ashape_[-1])
                self.quantizer_A.find_params(A, weight=self.pertoken_q)
                qA = quantize(
                    A, self.quantizer_A.scale, self.quantizer_A.zero, self.quantizer_A.maxq, self.sym_q
                ).to(A.dtype)
                qA = qA.reshape(Ashape_)
                # Quantizing B
                B = B.transpose(-1,-2)
                Bshape_ = B.shape
                B = B.reshape(-1, Bshape_[-1])
                self.quantizer_B.find_params(B, weight=self.pertoken_k)
                qB = quantize(
                    B, self.quantizer_B.scale, self.quantizer_B.zero, self.quantizer_B.maxq, self.sym_k
                ).to(B.dtype)
                qB = qB.reshape(Bshape_)
                qB = qB.transpose(-1,-2)
        else:
            qA = A
            qB = B
        return qA @ qB
    
class ActQuantMatMul2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.perchannel_s = not args.s_per_tensor
        self.pertoken_s = args.s_per_token
        self.groupsize_s = args.groupsize_s
        assert self.groupsize_s < 0, 'Group-wise quantization on multi-head activation is not supported yet'
        self.bit_s = args.bits_s
        self.sym_s = args.sym_s
        self.perchannel_v = not args.v_per_tensor
        self.pertoken_v = args.v_per_token
        self.groupsize_v = args.groupsize_v
        assert self.groupsize_v < 0, 'Group-wise quantization on multi-head activation is not supported yet'
        self.bit_v = args.bits_v
        self.sym_v = args.sym_v
        self.quantizer_A = Quantizer()
        self.quantizer_A.configure(
            self.bit_s, self.perchannel_s, sym=self.sym_s, mse=False
        )
        self.quantizer_B = Quantizer()
        self.quantizer_B.configure(
            self.bit_v, self.perchannel_v, sym=self.sym_v, mse=False
        )

    def forward(self, A, B): # [b, h, s, d]
        if (self.bit_s or self.bit_v) < 16: #change to < 16 for original
            if (self.bit_v) >= 16:  #change to => 16 for original
                # Quantizing A
                Ashape_ = A.shape
                A = A.reshape(-1, Ashape_[-1])
                self.quantizer_A.find_params(A, weight=self.pertoken_s)
                qA = quantize(
                    A, self.quantizer_A.scale, self.quantizer_A.zero, self.quantizer_A.maxq, self.sym_s
                ).to(A.dtype)
                qA = qA.reshape(Ashape_)
                qB = B
            elif (self.bit_s) >= 16: #change to => 16 for original
                # Quantizing B
                qA = A
                B = B.transpose(-1,-2)
                Bshape_ = B.shape
                B = B.reshape(-1, Bshape_[-1])
                self.quantizer_B.find_params(B, weight=self.pertoken_v)
                qB = quantize(
                    B, self.quantizer_B.scale, self.quantizer_B.zero, self.quantizer_B.maxq, self.sym_v
                ).to(B.dtype)
                qB = qB.reshape(Bshape_)
                qB = qB.transpose(-1,-2)
            else:
                # Quantizing A
                Ashape_ = A.shape
                A = A.reshape(-1, Ashape_[-1])
                self.quantizer_A.find_params(A, weight=self.pertoken_s)
                qA = quantize(
                    A, self.quantizer_A.scale, self.quantizer_A.zero, self.quantizer_A.maxq, self.sym_s
                ).to(A.dtype)
                qA = qA.reshape(Ashape_)
                # Quantizing B
                B = B.transpose(-1,-2)
                Bshape_ = B.shape
                B = B.reshape(-1, Bshape_[-1])
                self.quantizer_B.find_params(B, weight=self.pertoken_v)
                qB = quantize(
                    B, self.quantizer_B.scale, self.quantizer_B.zero, self.quantizer_B.maxq, self.sym_v
                ).to(B.dtype)
                qB = qB.reshape(Bshape_)
                qB = qB.transpose(-1,-2)
        else:
            qA = A
            qB = B
        return qA @ qB

def add_act_quant(model, args):
    from transformers.models.llama.modeling_llama import LlamaAttention
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            setattr(module, "matmul1", ActQuantMatMul1(args))
            setattr(module, "matmul2", ActQuantMatMul2(args))
            module.forward = MethodType(attention.llama_attn_forward, module)
            
    wrapped_modules={}
    module_dict={}
    it=[(name,m) for name,m in model.named_modules()]
    #logging.info('Add quantized modules for activation')
    for name,m in it:
        module_dict[name]=m
        idx=name.rfind('.')
        if idx==-1:
            idx=0
        father_name=name[:idx]
        if father_name in module_dict:
            father_module=module_dict[father_name]
        else:
            raise RuntimeError(f"father module {father_name} not found")
        if isinstance(m,nn.Linear) and 'head' not in name:
            idx = idx+1 if idx != 0 else idx
            new_m = ActQuantLinear(m.in_features,m.out_features,m.bias is not None,args=args)
            new_m.weight.data=m.weight.data
            new_m.bias=m.bias
            replace_m=new_m
            wrapped_modules[name] = new_m
            setattr(father_module,name[idx:],replace_m)

