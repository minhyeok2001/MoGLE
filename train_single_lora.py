import os
import argparse
import torch
import torch.nn as nn
import sentencepiece as spm
import json

from tqdm import tqdm
from typing import Optional, Tuple


import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init

from llama3.llama.single_LORA_model import TransformerWithSingleLoRA
from llama3.llama.multi_LORA_model import TransformerWithMoLE 
from llama3.llama.utils import ModelArgs

from utils.utils import init_model_parallel_if_needed
from utils.loss import LLMLoss


"""
Single Lora 학습 FLOW
1. 우선 데이터는 요청드린대로, (seq,cls)로 들어온다고 가정.
2. cls에 맞게 -> cls 로 자르는건 그냥 전처리부분에 이어서 하는게 좋을듯. 그래야 여려명이서 parallel하게 가능. 우선 입력으로 category를 받도록 설정하기
3. pretrained 는 https://www.llama.com/llama-downloads/ 여기 참고해서 받아오는것으로. huggingface꺼로는 못받아옴. ( 구조가 좀 다름 )
4. 

주의점
1. pretrain 잘 받아와지는지
2. lr, optimizer는 인터넷에 라마 파인튜닝한거 참고해서 가져오기

"""

def run(args):
    
    category = args.category
    base_path = args.base_path
    
    ckpt_path = os.path.join(base_path,"consolidated.00.pth")
    json_path = os.path.join(base_path,"params.json")
    
    ## 여기서 전처리코드 한번 거치고, 거기서 args.category 받아서 해당하는 데이터만 가져오면 좋을듯

    train_set = [torch.randint(0,100, (32,)) for _ in range(8)]
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=2,num_workers=4,shuffle=False)
    
    with open(json_path, "r") as f:
        cfg = json.load(f)
    vocab_size = cfg["vocab_size"]
    
    model_args = ModelArgs(vocab_size=vocab_size)
    model = TransformerWithSingleLoRA(model_args).to("cuda")
    
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    print("=== requires_grad 확인하기 ===")
    for name, param in model.named_parameters():
        print(f"{name}  requires_grad={param.requires_grad}")


    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=1e-5)
    
    loss_ft = LLMLoss()
    model.train()
    
    ## 이거 inference시에만 영향있는 인자라고 함 auto regressive하게 할때 ...
    start_pos = 0 
    
    for seq in tqdm(train_loader):
        seq = seq.to("cuda")
        
        logits = model(seq, start_pos)

        loss = loss_ft(logits, seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ## LoRA만 떼어다가 저장하기.
    os.makedirs(category, exist_ok=True)
    lora_path = os.path.join(f"{category}","lora_only.pt")
    
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state,lora_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category",type=str,required=True)
    parser.add_argument("--base_path",type=str,default="/root/.llama/checkpoints/Llama3.1-8B-Instruct") # 추후에 colab에서 돌려보고 기본 path 추가
    
    args = parser.parse_args()
    
    init_model_parallel_if_needed()
    run(args)
    torch.distributed.destroy_process_group()
    


