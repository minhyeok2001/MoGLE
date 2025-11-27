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
from dataset import SimpleTextDataset, get_dummy_texts


def run(args):
    
    wandb.init(
        project="Single_LoRA",
        name=args.genre,
        config={
            "genre": args.genre,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
        }
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

    print("=== Lora 주입중... ===")
    base_model = inject_single_lora(
        base_model,
        target_modules=args.target_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    model = base_model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    for m in model.modules():
        if isinstance(m, SingleLoraLinear):
            m.lora_A.weight.requires_grad = True
            m.lora_B.weight.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable LoRA: {trainable:,}")

    texts = get_dummy_texts()
    train_texts, val_texts = train_test_split(
        texts, test_size=0.1, shuffle=True, random_state=42
    )

    train_dataset = SimpleTextDataset(train_texts, tokenizer, max_len=args.max_len)
    val_dataset = SimpleTextDataset(val_texts, tokenizer, max_len=args.max_len)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    
    wandb.watch(model, log="all")

    num_epochs = args.epochs
    model.train()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_steps = 0
        for batch in tqdm(enumerate(train_dataloader)):
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
            loss = out.loss
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
                loss = out.loss
                
                total_val_loss += loss.item() 
                total_val_steps += 1

        avg_val_loss = total_val_loss / total_val_steps
        print(f"[epoch {epoch}] val_loss={avg_val_loss:.4f}")
        
        wandb.log(
            {
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "epoch": epoch,
            }
        )
        
    os.makedirs(args.lora_base_path, exist_ok=True)
    save_path = os.path.join(args.lora_base_path, f"expert_{args.genre}.ckpt")

    ## LORA만 빼내기
    full_sd = model.state_dict()
    lora_sd = {
        k: v.cpu()
        for k, v in full_sd.items()
        if ("lora_A" in k) or ("lora_B" in k)
    }

    torch.save(lora_sd, save_path)
    print(f"LoRA only : {save_path}")
    
    wandb.finish()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    ## 얘는 실행할때 --genre로 여러개 주고 그대로 ckpt 저장하는식으로

    parser.add_argument("--genre", type=str, required=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules",type=str,default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--lora_base_path", type=str, default="/checkpoints")

    args = parser.parse_args()
    args.target_modules = args.target_modules.split(",")

    run(args)