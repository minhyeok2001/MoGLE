import torch
import torch.nn as nn
import torch.nn.functional as F

class LLMLoss(nn.Module):
    def __init__(self, pad_id=128001): ## 찾아보니, padding token id가 128001임
        super().__init__()
        self.pad_id = pad_id

    def forward(self, logits, input_ids):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.pad_id
        )
        return loss