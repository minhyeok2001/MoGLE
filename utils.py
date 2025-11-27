import os
import torch
import torch.nn as nn

from .module import MultiExpertLoraLinear, SingleLoraLinear
import mask_state

def capture_attention_mask(module, args, kwargs):
    attn = None
    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
        attn = kwargs["attention_mask"]
    elif len(args) >= 2:
        attn = args[1]
    mask_state.CURRENT_ATTENTION_MASK = attn

def get_parent_module(model: nn.Module, module_name: str):
    ## 이거 model.layers.3.self_attn.q_proj 같은거에서 마지막 q_proj만 꺼내기 위한 함수임
    ## .으로 나누기 -> 있는애들은 부모까지 이름만 가져오기. 자식은 그 우리가 따로 설정해주는 target module에서 가져옴
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:  
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_single_lora(
    model: nn.Module,
    target_modules,
    r: int,
    lora_alpha: int,
):
    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if not any(t in name for t in target_modules):
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            continue

        parent, child_name = get_parent_module(model, name)
        new_mod = SingleLoraLinear(
            base_layer=module,
            r=r,
            lora_alpha=lora_alpha,
        )
        setattr(parent, child_name, new_mod)

    return model


def inject_layerwise_lora(
    model: nn.Module,
    target_modules,
    num_experts,
    r,
    lora_alpha,
    expert_ckpt_paths=None, 
):
    if expert_ckpt_paths is not None:
        assert len(expert_ckpt_paths) == num_experts
        expert_sds = [torch.load(p, map_location="cpu") for p in expert_ckpt_paths]
    else:
        print("@@@@@@@@@@@@@@@@@@ Expert unloaded!! @@@@@@@@@@@@@@@@@@")
        expert_sds = None

    for p in model.parameters():
        p.requires_grad = False

    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if not any(t in name for t in target_modules):
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            continue

        parent, child_name = get_parent_module(model, name)

        new_mod = MultiExpertLoraLinear(
            base_layer=module,
            num_experts=num_experts,
            r=r,
            lora_alpha=lora_alpha,
        )

        ## 여기서 로라 로드 
        ## sd = torch.load("lora_expert0.pth"), sds = [sd0, sd1, sd2] 이런식으로 사용하기
        if expert_sds is not None:
            for e in range(num_experts):
                sd = expert_sds[e]
                key_A = f"{name}.lora_A.weight"
                key_B = f"{name}.lora_B.weight"
                if key_A in sd and key_B in sd:
                    with torch.no_grad():
                        new_mod.lora_A[e].weight.copy_(sd[key_A])
                        new_mod.lora_B[e].weight.copy_(sd[key_B])
                else:
                    raise RuntimeError("@@@@@@@@@@@@@@@@@@모듈 개수 안맞음@@@@@@@@@@@@@@@@@@")

        setattr(parent, child_name, new_mod)

    return model