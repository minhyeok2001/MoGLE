import os
import math
import torch
import torch.nn as nn
import argparse
import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from utils import *
from dataset import GenreStoryDataset


def run(args):

    wandb.init(
        project="MoLE", 
        config={
            "genre": args.genre,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "lora_base_path": args.lora_base_path,
            "balance_weight": args.balance_weight,
        },
    )
    if not torch.cuda.is_available():
        raise RuntimeError("무조건 CUDA로 하셔야함")
    
    device = "cuda" 
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    print("=== 토크나이저 로딩... ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    print("=== BnB 로딩... ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("=== 4비트 모델 로딩... ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    num_experts = len(args.genre)

    paths = []
    for genre in args.genre:
        p = os.path.join(args.lora_base_path, f"expert_{genre}.ckpt")
        paths.append(p)
        
        
    print("=== Lora 주입중... ===")
    base_model = inject_layerwise_lora(
        base_model,
        target_modules=args.target_modules,
        num_experts=num_experts,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        expert_ckpt_paths=paths,
    )

    base_model.register_forward_pre_hook(capture_attention_mask, with_kwargs=True)
    
    model = base_model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    for m in model.modules():
        if isinstance(m, MultiExpertLoraLinear):
            m.gate.weight.requires_grad = True
            if m.gate.bias is not None:
                m.gate.bias.requires_grad = True
            if hasattr(m, "tau"):
                m.tau.requires_grad = True
                
    total = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable - Gate, Tau: {trainable_after:,}")
    
    train_dataset = GenreStoryDataset(tokenizer=tokenizer,genres=None, max_len=args.max_len,train_flag=True)
    val_dataset = GenreStoryDataset(tokenizer=tokenizer,genres=None,max_len=args.max_len,train_flag=False)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    
    num_epochs = args.epochs
    model.train()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_steps = 0
        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optim.zero_grad()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            llm_loss = out.loss
            balance_loss = compute_balance_loss(model)
            
            loss = llm_loss + args.balance_weight * balance_loss
            loss.backward()
            optim.step()
            
            total_train_loss += loss.item() 
            total_train_steps += 1
            
        avg_train_loss = total_train_loss / total_train_steps
        print(f"[epoch {epoch}] train_loss={avg_train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            total_val_steps = 0
            for batch in tqdm(val_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                llm_loss = out.loss
                balance_loss = compute_balance_loss(model)      
                loss = llm_loss + args.balance_weight * balance_loss

                total_val_loss += loss.item() 
                total_val_steps += 1

        avg_val_loss = total_val_loss / total_val_steps
        print(f"[epoch {epoch}] val_loss={avg_val_loss:.4f}")
        
        gate_info = {}
        for name, module in model.named_modules():
            if isinstance(module, MultiExpertLoraLinear):
                w = module.last_gate_weights
                if w is None:
                    continue

                w_mean = w.mean(dim=0).detach().cpu()
                w_mean_list = [round(x.item(), 4) for x in w_mean]
                print(f"[module] {name}")
                print(f"   tau: {module.tau.item():.4f}")
                print(f"   gate_mean: {w_mean_list}")
                print("-"*60)

                for i, v in enumerate(w_mean):
                    gate_info[f"gate/{name}/expert_{i}"] = v.item()
                gate_info[f"gate/{name}/tau"] = module.tau.item()

        log_dict = {
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "epoch": epoch,
        }
        log_dict.update(gate_info)

        wandb.log(log_dict)


    save_dir = os.path.join(args.lora_base_path, "gate_ckpts")
    os.makedirs(save_dir, exist_ok=True)

    gate_sd = {}

    for name, module in model.named_modules():
        if isinstance(module, MultiExpertLoraLinear):
            if hasattr(module, "gate"):
                gate_sd[f"{name}.gate.weight"] = module.gate.weight.cpu().clone()
                if module.gate.bias is not None:
                    gate_sd[f"{name}.gate.bias"] = module.gate.bias.cpu().clone()
            if hasattr(module, "tau"):
                gate_sd[f"{name}.tau"] = module.tau.detach().cpu().clone()

    save_path = os.path.join(
        save_dir,
        f"mole_gate_{args.balance_weight}.ckpt"
    )
    torch.save(gate_sd, save_path)
    print(f"[저장 완료] Gate/Tau ckpt: {save_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--genre",type=str,required=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules",type=str,default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lora_base_path",type=str,default="/checkpoints")
    parser.add_argument("--balance_weight",type=float,default=0.5)

    args = parser.parse_args()
    args.target_modules = args.target_modules.split(",")
    args.genre = args.genre.split(",")

    run(args)
