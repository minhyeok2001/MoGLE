import torch
import torch.nn as nn
from llama3.llama.single_LORA_model import TransformerWithSingleLoRA
from llama3.llama.multi_LORA_model import TransformerWithMoLE 
from llama3.llama.utils import ModelArgs
import fairscale.nn.model_parallel.initialize as fs_init

def init_model_parallel_if_needed(mp_size=1):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    if fs_init.get_model_parallel_world_size() != mp_size:
        fs_init.initialize_model_parallel(mp_size)

init_model_parallel_if_needed(1)

args = ModelArgs(
    dim=64,             
    n_layers=1,
    n_heads=4,
    n_kv_heads=4,
    vocab_size=100,
    multiple_of=16,
    max_batch_size=4,
    max_seq_len=32,
)
args.num_experts = 3
args.r = 4
args.lora_alpha = 8
model = TransformerWithSingleLoRA(params=args)
model2 = TransformerWithMoLE(params=args)

B, T = 2, 5
x = torch.randn(B, T, args.dim)

start_pos = 0
out = model(x,start_pos)

out2 = model2(x,start_pos)

print("input shape :", x.shape)

print("====== single-lora ======")
print("output shape:", out.shape)


print("====== multi-lora ======")
print("output shape:", out2.shape)