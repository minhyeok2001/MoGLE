
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

from .utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoLEModule(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        num_experts: int,
        r: int = 16,
        lora_alpha: int = 32,
        gate_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.base = base_layer
        self.num_experts = num_experts
        self.r = r
        self.scaling = lora_alpha / r
        self.gate_norm_eps = gate_norm_eps

        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.lora_A = nn.ModuleList([nn.Linear(in_dim, r, bias=False) for _ in range(num_experts)])
        self.lora_B = nn.ModuleList([nn.Linear(r, out_dim, bias=False) for _ in range(num_experts)])

        self.norm = None
        self.e_proj = None
        self.tau = nn.Parameter(torch.tensor(1.0))

    ## 이거 런타임 시점까지 dim을 알 수 없으므로, lazy하게 진행
    def _lazy_init_e(self, flat_dim: int, device):
        self.e_proj = nn.Linear(flat_dim, self.num_experts, bias=False).to(device)
        
    ## norm도 마찬가지
    def _lazy_init_norm(self, last_dim, device):
        self.norm = nn.LayerNorm(last_dim, eps=self.gate_norm_eps).to(device)

    def forward(self, x):
        base_out = self.base(x)

        expert_outs = []
        for k in range(self.num_experts):
            lora_k = self.lora_B[k](self.lora_A[k](x)) * self.scaling
            E_k = base_out + lora_k
            expert_outs.append(E_k)
            
        E = torch.stack(expert_outs, dim=0)

        E_cat = E.permute(1, 2, 0, 3).contiguous().view(x.size(0), x.size(1), -1) # 차원은 아마 B T C
        
        if self.norm is None:
            self._lazy_init_norm(E_cat.size(-1), E_cat.device)
            
        E_cat = self.norm(E_cat)

        flat = E_cat.view(x.size(0), -1)

        if self.e_proj is None:
            self._lazy_init_e(flat.size(-1), flat.device)
            
        eps = self.e_proj(flat)

        gates = F.softmax(eps / self.tau.clamp_min(1e-6), dim=-1)

        g = gates.permute(1,0)[:, :, None, None]
        out = (E * g).sum(dim=0) 

        return out
    

class AttentionWithMoLE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        wq_base = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        wk_base = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        wv_base = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.wq = MoLEModule(
            wq_base, num_experts=args.num_experts,
            r=args.r, lora_alpha=args.lora_alpha,
            gate_norm_eps=args.gate_norm_eps
        )
        self.wk = MoLEModule(
            wk_base, num_experts=args.num_experts,
            r=args.r, lora_alpha=args.lora_alpha,
            gate_norm_eps=args.gate_norm_eps
        )
        self.wv = MoLEModule(
            wv_base, num_experts=args.num_experts,
            r=args.r, lora_alpha=args.lora_alpha,
            gate_norm_eps=args.gate_norm_eps
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForwardWithMoLE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        num_experts : int = 10,
        r : int = 16,
        lora_alpha : int = 32,
        gate_norm_eps : float = 1e-6
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        w1_base = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        w2_base = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        w3_base = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

        self.w1 = MoLEModule(w1_base, num_experts, r, lora_alpha, gate_norm_eps)
        self.w2 = MoLEModule(w2_base, num_experts, r, lora_alpha, gate_norm_eps)
        self.w3 = MoLEModule(w3_base, num_experts, r, lora_alpha, gate_norm_eps)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlockWithMoLE(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = AttentionWithMoLE(args)
        self.feed_forward = FeedForwardWithMoLE(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            num_experts=args.num_experts,
            r=args.r,
            lora_alpha=args.lora_alpha
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    

class TransformerWithMoLE(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlockWithMoLE(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
