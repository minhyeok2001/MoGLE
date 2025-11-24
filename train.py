import torch
import torch.nn as nn
from llama3.llama.single_LORA_model import TransformerWithSingleLoRA
from llama3.llama.multi_LORA_model import TransformerWithMoLE 
from llama3.llama.utils import ModelArgs

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
model = TransformerWithSingleLoRA(args=args)
model2 = TransformerWithMoLE(args=args)

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