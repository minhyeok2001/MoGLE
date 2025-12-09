import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class GenreStoryDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        csv_path="dataset.csv",
        max_len=2048,
        genres=None,
        train_flag=True,
        val_ratio=0.1,
        seed=42,
        training_target="lora",
    ):
        if genres == "all":
            genres = None
            
        df = pd.read_csv(csv_path)
        length_df = len(df)
        cut = length_df // 3

        if training_target == "lora":
            df = df.iloc[: 2 * cut]
        elif training_target == "mole":
            df = df.iloc[2 * cut :]
        elif training_target == "all":
            df = df
        else:
            raise RuntimeError("lora / mole 둘중 선택 !!")

        if genres is not None:
            if isinstance(genres, str):
                genres = [genres]
            df = df[df["genre"].isin(genres)].reset_index(drop=True)

        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            shuffle=True,
            random_state=seed,
        )

        if train_flag:
            df = train_df.reset_index(drop=True)
        else:
            df = val_df.reset_index(drop=True)

        self.genres = df["genre"].tolist()
        self.sources = df["source"].tolist()
        stories = df["story"].tolist()

        enc = tokenizer(
            stories,
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
            "genre": self.genres[idx],
            "source": self.sources[idx],
        }