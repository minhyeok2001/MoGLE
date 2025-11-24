import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


model = AutoModelForCausalLM.from_pretrained(model_id, token=token, device_map="auto")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
tokenizer.pad_token = tokenizer.eos_token
