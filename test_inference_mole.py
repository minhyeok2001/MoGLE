import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import inject_layerwise_lora, MultiExpertLoraLinear, capture_attention_mask

parser = argparse.ArgumentParser()
parser.add_argument("--gate_weight", type=str, required=True)
args = parser.parse_args()

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_BASE_PATH = "/checkpoints"
GATE_CKPT_PATH = f"/checkpoints/gate_ckpts/mole_{args.gate_weight}.ckpt"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")
device = "cuda"

expert_files = [
    f for f in os.listdir(LORA_BASE_PATH)
    if f.startswith("expert_") and f.endswith(".ckpt")
]

def extract_genre(fname):
    return fname[len("expert_") : -len(".ckpt")]

GENRES = sorted(extract_genre(f) for f in expert_files)
print("genres:", GENRES)

expert_ckpt_paths = [
    os.path.join(LORA_BASE_PATH, f"expert_{g}.ckpt") for g in GENRES
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

base_model.register_forward_pre_hook(capture_attention_mask, with_kwargs=True)

target_modules = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj"]
lora_r = 8
lora_alpha = 16

model = inject_layerwise_lora(
    base_model,
    target_modules=target_modules,
    num_experts=len(GENRES),
    r=lora_r,
    lora_alpha=lora_alpha,
    expert_ckpt_paths=expert_ckpt_paths,
)

gate_sd = torch.load(GATE_CKPT_PATH, map_location="cpu")
model.load_state_dict(gate_sd, strict=False)

model.to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

@torch.no_grad()
def generate_with_mole(prompt, max_new_tokens=128, temperature=0.7, top_p=0.9):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

test_prompt = (
    "You are a writing assistant specializing in Sci-Fi. "
    "User: Write a short opening paragraph in a Sci-Fi style.\n"
    "Assistant:"
)

print(generate_with_mole(test_prompt, max_new_tokens=200))