import math
import torch
import torch.nn as nn
import mask_state 

class SingleLoraLinear(nn.Module):
    def __init__(self, base_layer, r, lora_alpha):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.r = r
        self.scaling = lora_alpha / r

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        base_out = self.base_layer(x_2d)
        base_dtype = base_out.dtype

        x_lora = x_2d.to(self.lora_A.weight.dtype)
        delta = self.lora_B(self.lora_A(x_lora))
        delta = delta.to(base_dtype)

        out = base_out + self.scaling * delta
        out = out.view(*orig_shape[:-1], self.out_features)
        return out

class MultiExpertLoraLinear(nn.Module):
    def __init__(self, base_layer, num_experts, r, lora_alpha):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.num_experts = num_experts
        self.r = r
        self.scaling = lora_alpha / r

        gate_in_dim = self.out_features * self.num_experts
        self.gate = nn.Linear(gate_in_dim, num_experts)

        self.lora_A = nn.ModuleList([
            nn.Linear(self.in_features, r, bias=False) for _ in range(num_experts)
        ])
        self.lora_B = nn.ModuleList([
            nn.Linear(r, self.out_features, bias=False) for _ in range(num_experts)
        ])

        ## 아 이게 없으면 초기 값이 너무 이상하게 나옴 !!
        for A, B in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
            nn.init.zeros_(B.weight)

        self.tau = nn.Parameter(torch.tensor(1.0))
                
        ## 디버깅용
        self.last_gate_weights = None

    def forward(self, x):
        orig_shape = x.shape
        B, T, D = x.shape
        x_2d = x.view(-1, self.in_features)

        base_out = self.base_layer(x_2d)
        base_dtype = base_out.dtype

        attn_mask = mask_state.CURRENT_ATTENTION_MASK
        if attn_mask is None:
            raise RuntimeError("마스크가 없어요~")
        attn_mask = attn_mask.to(x.device)
        mask = attn_mask.unsqueeze(-1)

        delta_list = []
        pooled_list = []

        for e in range(self.num_experts):
            A = self.lora_A[e]
            B_lin = self.lora_B[e]

            ## 일단 로라 거치고
            x_lora = x_2d.to(A.weight.dtype)
            delta_e = B_lin(A(x_lora))
            delta_e = delta_e.to(base_dtype)

            ## 지금 마스크가 B T 1꼴임 
            delta_e_3d = delta_e.view(B, T, self.out_features)
            mask_f = mask.to(delta_e_3d.dtype)

            ## 마스크 씌워서 패딩 없애고, 나머지중에 평균내기
            summed = (delta_e_3d * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1e-6)
            pooled_e = summed / denom

            delta_list.append(delta_e)
            pooled_list.append(pooled_e)

        ## 논문에서 말하는 normalize는 그냥 위에서 패딩하고 평균낸걸로 썜썜치고, tau랑 
        gate_in = torch.cat(pooled_list, dim=-1).to(torch.float32)
        gate_logits = self.gate(gate_in) / self.tau
        weights = torch.softmax(gate_logits, dim=-1)

        ## 이거 디버그용 & loss용으로 따로 빼둠
        self.last_gate_weights = weights

        w_expanded = weights.unsqueeze(1).expand(B, T, self.num_experts).reshape(-1, self.num_experts)

        delta_sum = torch.zeros_like(base_out, dtype=base_dtype)
        for e in range(self.num_experts):
            delta_e = delta_list[e]
            w_e = w_expanded[:, e].unsqueeze(1).to(base_dtype)
            delta_sum = delta_sum + w_e * delta_e

        out = base_out + self.scaling * delta_sum
        out = out.view(*orig_shape[:-1], self.out_features)
        return out