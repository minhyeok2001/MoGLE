import torch

import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init


## 이거 해야 Llama에서 사용하던 MP관련 에러 안남
def init_model_parallel_if_needed(mp_size=1):
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1
        )
    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(mp_size)

    torch.cuda.set_device(0)