import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attn_mask = enc["attention_mask"]

        self.labels = self.input_ids.clone()
        self.labels[self.attn_mask == 0] = -100

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "labels": self.labels[idx],
        }


def get_dummy_texts():
    base_texts = [
        "The wizard opened the ancient grimoire under the moonlight.",
        "A small dragon curled up by the fireplace, snoring softly.",
        "Modern data centers rely on thousands of interconnected servers.",
        "The neural network struggled to generalize from the tiny dataset.",
        "In a distant kingdom, magic and science coexisted uneasily.",
        "장르에 따라 문장의 리듬과 어휘 선택이 완전히 달라진다.",
        "이 도시는 마법사가 아닌 엔지니어가 세상을 움직이는 곳이다.",
        "서버 로그를 분석하자 예기치 못한 트래픽 패턴이 드러났다.",
        "용 사냥꾼은 오래된 전설 속 주문을 중얼거리며 숲을 걸었다.",
        "인공지능 모델은 점점 더 미묘한 스타일 차이를 포착하기 시작했다.",
    ]
    return base_texts * 10