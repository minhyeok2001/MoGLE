import torch
import torch.nn as nn
from llama3.llama.single_LORA_model import TransformerWithSingleLoRA
from llama3.llama.multi_LORA_model import TransformerWithMoLE 
from llama3.llama.utils import ModelArgs

import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init
import os
import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

#MP 설정 끄는 부분이 필요하다고함
def init_model_parallel_if_needed(mp_size=1):
    # 1) torch.distributed init 먼저!!!!
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1
        )

    # 2) model-parallel 그룹 초기화
    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(mp_size)

    torch.cuda.set_device(0)

init_model_parallel_if_needed()

args = ModelArgs(
    dim=64,             
    n_layers=1,
    n_heads=4,
    n_kv_heads=4,
    vocab_size=100,
    multiple_of=16,
    max_batch_size=4,
    max_seq_len=32,
    num_experts = 3,
    r = 4,
    lora_alpha = 8,
    gate_norm_eps = 1e-6
)

model = TransformerWithSingleLoRA(params=args)
model2 = TransformerWithMoLE(params=args)

B, T = 2, 5
vocab = args.vocab_size
x = torch.randint(0, vocab, (B, T), device="cuda")

start_pos = 0
out = model(x,start_pos)

out2 = model2(x,start_pos)

print("input shape :", x.shape)

print("====== single-lora ======")
print("output shape:", out.shape)


print("====== multi-lora ======")
print("output shape:", out2.shape)