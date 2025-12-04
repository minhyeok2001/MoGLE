import os
import math
import torch
import torch.nn as nn
import argparse
import wandb
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from utils import *
from sample_dataset import SimpleTextDataset, get_dummy_texts
from dataset import GenreStoryDataset


def run(args):

    wandb.init(
        project="Full_FT",
        name=args.genre,
        config={
            "genre": args.genre,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
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

    print("=== 풀파인튠용 모델 로딩 (bf16) ... ===")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,   
        device_map="auto",
    )

    for p in model.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable params: {trainable:,}")

    train_dataset = GenreStoryDataset(
        tokenizer=tokenizer,
        genres=args.genre,
        max_len=args.max_len,
        train_flag=True,
        training_target="all",  
    )
    val_dataset = GenreStoryDataset(
        tokenizer=tokenizer,
        genres=args.genre,
        max_len=args.max_len,
        train_flag=False,
        training_target="all",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("== dataset size ==")
    print(" train :", len(train_dataset))
    print(" val :", len(val_dataset))

    optim = torch.optim.AdamW(
        model.parameters(),
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

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch} [train]"):
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

        avg_train_loss = total_train_loss / max(total_train_steps, 1)
        print(f"[epoch {epoch}] train_loss={avg_train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            total_val_steps = 0

            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch} [val]"):
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

        avg_val_loss = total_val_loss / max(total_val_steps, 1)
        print(f"[epoch {epoch}] val_loss={avg_val_loss:.4f}")

        wandb.log(
            {
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "epoch": epoch,
            }
        )
        
    os.makedirs(args.save_dir, exist_ok=True)

    save_path = os.path.join(args.save_dir, f"ft_{args.genre}.ckpt")
    torch.save(model.state_dict(), save_path)

    print(f"Full FT saved to ckpt: {save_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_fullft")

    args = parser.parse_args()
    run(args)