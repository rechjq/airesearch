
from dataclasses import dataclass
from  typing import Optional,Tuple
import json

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads:int = 32
    n_kv_heads: Optional[int] =None
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float=1e-5
    max_seq_len:int = 2048
    dropout: float = 0.0
@dataclass
class LLamaTrainConfig:
    learning_rate: float= 3e-4
    weight_decay: float = 1e-1
    beta1:float = 0.9
    beta2:float = 0.95
    grad_clip: float = 1.0
    max_epoch: int = 1
    log_interval: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    decay_lr: bool = True
    min_lr: float=1e-5
    lr_decay_iters:int=80000
    warmup_iters:int = 1000
    device:str = "cuda"
    dtype:str = "float16"
    output:str = "out"
    backend:str = "nccl"
    best_val_loss:float = 1e9


class LLamaConfig(object):
    def __init__(self, **kwargs) -> None:
        ma = kwargs.pop("model",None)
        ta = kwargs.pop("train",None)
        self.modelArgs = ModelArgs() if ma is None else ModelArgs(**ma)
        self.trainArgs = LLamaTrainConfig() if ta is None else LLamaTrainConfig(**ta)
    @classmethod
    def from_pretrain(cls, config):
        with open(config) as f:
            data = json.load(f)
            return cls(**data)
if __name__ == '__main__':
    c=LLamaConfig.from_pretrain("config.json")
    print(c.__dict__)