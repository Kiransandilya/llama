from .generation import LLaMA
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer

from typing import Tuple
import json
import os
import sys
import time
import torch


from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from pathlib import Path

class Q:
    
    def __init__(self):
        self.ckpt_dir = '/raid/mpsych/llama/weights/7B/7B/'
        self.tokenizer_path = '/raid/mpsych/llama/weights/7B/tokenizer.model'
        self.temperature: float = 0.8
        self.top_p: float = 0.95
        self.max_seq_len: int = 512
        self.max_batch_size: int = 32

        os.environ['RANK'] = '0' # single gpu only
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '31337'
        local_rank, world_size = self.setup_model_parallel()
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        self.generator = self.load(
            self.ckpt_dir, 
            self.tokenizer_path, 
            local_rank, 
            world_size, 
            self.max_seq_len, 
            self.max_batch_size
        )

    def setup_model_parallel(self) -> Tuple[int, int]:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))

        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)
        return local_rank, world_size
        
    def load(self,
        ckpt_dir: str,
        tokenizer_path: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        max_batch_size: int,
    ) -> LLaMA:
        
        start_time = time.time()
        
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        print(ckpt_dir, checkpoints)
        
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        
        ckpt_path = checkpoints[local_rank]
        print("Loading")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

        generator = LLaMA(model, tokenizer)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return generator
    
    def ask(self, what, max_gen_len=256, temperature=0.8, top_p=0.95):
        res = self.generator.generate([what], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        
        return res
            