import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import inject_single_lora, SingleLoraLinear

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
GENRE = "Sci-Fi"
CKPT_PATH = f"/expert_{GENRE}.ckpt"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

device = "cuda"

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

target_modules = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj"]
lora_r = 8
lora_alpha = 16

model = inject_single_lora(
    base_model,
    target_modules=target_modules,
    r=lora_r,
    lora_alpha=lora_alpha,
)

lora_sd = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(lora_sd, strict=False)

model.eval()
for p in model.parameters():
    p.requires_grad = False

@torch.no_grad()
def generate_with_lora(prompt, max_new_tokens=128, temperature=0.7, top_p=0.9):
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

test_prompt = f"You are a writing assistant specializing in {GENRE} stories.\nUser: Write a short opening paragraph in the style of {GENRE}.\nAssistant:"
print(generate_with_lora(test_prompt, max_new_tokens=200))