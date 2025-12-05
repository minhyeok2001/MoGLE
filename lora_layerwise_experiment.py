# layerwise_lora_generation.py

import os
import re
import argparse

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import inject_single_lora
from module import SingleLoraLinear
import wandb


MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def debug_print_lora_layers(model: nn.Module, start_layer: int, end_layer: int):
    """
    현재 모델에서 어떤 레이어에 LoRA가 살아있는지 요약해서 프린트해주는 함수.
    zero_out_lora_outside_layer_range 호출 이후에 쓰면 됨.
    """
    layer_stats = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, SingleLoraLinear):
                continue

            layer_idx = get_layer_idx_from_name(name)
            if layer_idx is None:
                continue

            # norm이 0이면 완전히 꺼진 상태, 0보다 크면 LoRA가 살아있음
            a_norm = module.lora_A.weight.abs().sum().item()
            b_norm = module.lora_B.weight.abs().sum().item()
            has_lora = (a_norm > 0) or (b_norm > 0)

            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = {"cnt": 0, "active": 0}
            layer_stats[layer_idx]["cnt"] += 1
            if has_lora:
                layer_stats[layer_idx]["active"] += 1

    print("\n=== LoRA layer debug ===")
    print(f"intended active range: [{start_layer}, {end_layer})")
    active_layers = []
    for l in sorted(layer_stats.keys()):
        info = layer_stats[l]
        print(
            f"layer {l:2d}: modules={info['cnt']:2d}, "
            f"active_lora_modules={info['active']:2d}"
        )
        if info["active"] > 0:
            active_layers.append(l)

    print("=> actual active layers:", active_layers)
    print("=====================================\n")
    
    
def preprocess_csv(csv_path, split_type="all"):
    df = pd.read_csv(csv_path)

    n = len(df)
    if split_type == "A":
        df = df.iloc[: n // 2].reset_index(drop=True)
    elif split_type == "B":
        df = df.iloc[n // 2 :].reset_index(drop=True)
    elif split_type =="all":
        df = df
    else:
        raise RuntimeError("유효한 TYPE 주세요")


    def build_prompts(row):
        u1 = str(row["사람1"])
        a1 = str(row["AI1"])
        u2 = str(row["사람2"])
        a2 = str(row["AI2"])
        u3 = str(row["사람3"])

        GT1 = str(row["GT1"])
        GT2 = str(row["GT2"])
        GT3 = str(row["GT3"])

        prompt_v1 = f"user: {u1}"
        target_v1 = GT1

        prompt_v2 = "\n".join([
            f"user: {u1}",
            f"assistant: {a1}",
            f"user: {u2}",
        ])
        target_v2 = prompt_v2 + "\nassistant: " + GT2

        prompt_v3 = "\n".join([
            f"user: {u1}",
            f"assistant: {a1}",
            f"user: {u2}",
            f"assistant: {a2}",
            f"user: {u3}",
            
        ])
        target_v3 = prompt_v3 + "\nassistant: " + GT3

        return pd.Series({
            "prompt_v1": prompt_v1,
            "prompt_v2": prompt_v2,
            "prompt_v3": prompt_v3,
            "target_v1": target_v1,
            "target_v2": target_v2,
            "target_v3": target_v3,
        })

    prompts = df.apply(build_prompts, axis=1)
    df = pd.concat([df, prompts], axis=1)
    return df


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model_4bit():
    if not torch.cuda.is_available():
        raise RuntimeError("무조건 CUDA로 하셔야함")

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
    return base_model

def get_layer_idx_from_name(name: str):
    m = LAYER_RE.search(name)
    if m is None:
        return None
    return int(m.group(1))


def load_single_lora_ckpt(model: nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, SingleLoraLinear):
                key_A = f"{name}.lora_A.weight"
                key_B = f"{name}.lora_B.weight"
                if key_A in sd and key_B in sd:
                    module.lora_A.weight.copy_(sd[key_A])
                    module.lora_B.weight.copy_(sd[key_B])
                else:
                    raise RuntimeError(f"LoRA key mismatch at {name}")


def zero_out_lora_outside_layer_range(
    model: nn.Module,
    start_layer: int,
    end_layer: int,
):
    """
    [start_layer, end_layer) 범위 밖의 레이어에 있는 SingleLoraLinear의
    lora_A/B weight를 0으로 만들어서 '해제'하는 함수.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, SingleLoraLinear):
                continue
            layer_idx = get_layer_idx_from_name(name)
            if layer_idx is None:
                continue
            if not (start_layer <= layer_idx < end_layer):
                module.lora_A.weight.zero_()
                module.lora_B.weight.zero_()


            
@torch.no_grad()
def generate_with_model(prompt_list, tokenizer, model, device="cuda", max_new_tokens=512):
    full_outputs = []
    model_only_outputs = []

    def cut_before_user(text: str):
        lines = text.splitlines()
        safe = []
        for line in lines:
            if line.strip().startswith("user:"):
                break
            elif line.strip().startswith("assistant:"):
                break
            safe.append(line)
        return "\n".join(safe).rstrip()

    for p in prompt_list:

        inputs = tokenizer(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        input_ids = inputs["input_ids"]

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,  
        )

        full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        if full_text.startswith(prompt_text):
            raw_model_part = full_text[len(prompt_text):].lstrip()
        else:
            gen_ids = out_ids[0][input_ids.shape[1]:]
            raw_model_part = tokenizer.decode(gen_ids, skip_special_tokens=True).lstrip()

        # ← 여기서 user: 등장하면 그 이전까지 잘라냄
        model_only_text = cut_before_user(raw_model_part)

        full_outputs.append(full_text)
        model_only_outputs.append(model_only_text)

        print("\n" + "="*80)
        print("[INPUT PROMPT]")
        print(p)
        print("\n[MODEL ONLY OUTPUT]")
        print(model_only_text)
        print("="*80 + "\n")

    return full_outputs, model_only_outputs

def run(args):
    
    wandb.init(
        project="MoLE_layerwise",
        config={
            "layer_slices": args.layer_slices
        },
    )
        
    device = "cuda"
    
    df = preprocess_csv("eval.csv", args.type)
    genre_list = df["genre"].tolist()
    prompt_list = df["prompt_v2"].tolist()
    gt_list = df["GT2"].tolist()
    gt_cum_list = df["target_v2"].tolist()

    print("\n===== Prompt v2 (sample) =====")
    for i in range(min(3, len(prompt_list))):
        print(f"[{i}] {prompt_list[i]}\n")

    tokenizer = load_tokenizer()

    tmp_model = load_base_model_4bit()
    num_layers = len(tmp_model.model.layers)
    del tmp_model
    print(f"Total transformer layers: {num_layers}")

    slice_specs = []
    for token in args.layer_slices.split(","):
        s, e = token.split("-")
        slice_specs.append((int(s), int(e)))

    for (s_pct, e_pct) in slice_specs:
        print(f"\n=== Slice {s_pct}-{e_pct}% ===")

        base_model = load_base_model_4bit()
        base_model = inject_single_lora(
            base_model,
            target_modules=args.target_modules,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        load_single_lora_ckpt(base_model, "/checkpoints/expert_all_no_scale.ckpt")

        start_layer = int(num_layers * (s_pct / 100.0))
        end_layer = int(num_layers * (e_pct / 100.0))
        print(f" -> using layers [{start_layer}, {end_layer}) for LoRA")

        zero_out_lora_outside_layer_range(
            base_model,
            start_layer=start_layer,
            end_layer=end_layer,
        )
        
        debug_print_lora_layers(base_model, start_layer, end_layer)

        model = base_model.to(device)
        model.eval()
        
        system_prefix = (
            "system: You are an AI assistant. "
            "Continue the conversation ONLY as 'assistant:'. "
            "Never write lines starting with 'user:'."
        )

        gen_prompt_list = []
        for p in prompt_list:
            full_p = system_prefix + "\n" + p + "\nassistant:"
            gen_prompt_list.append(full_p)

        print("GEN PROMPT LIST :", gen_prompt_list[0])
        
        generate_with_model(
            gen_prompt_list,
            tokenizer,
            model,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--layer_slices",
        type=str,
        default="0-20,20-40,40-60,60-80,80-100",
        help="퍼센트 구간 리스트. 예: 0-20,20-40,40-60,60-80,80-100",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )

    args = parser.parse_args()
    args.target_modules = args.target_modules.split(",")

    run(args)