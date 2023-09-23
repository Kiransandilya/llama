import torch
import json
import argparse
import threading

from accelerate import init_empty_weights, infer_auto_device_map
import transformers
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers import StoppingCriteria, StoppingCriteriaList
from loguru import logger
from typing import List, Union


def get_device_map(model_name, device, do_int8):
    if device == "a100-40g":
        return "auto"

    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    d = {0: "18GiB"}
    for i in range(1, 6):
        d[i] = "26GiB"
    device_map = infer_auto_device_map(
        model, max_memory=d, dtype=torch.int8 if do_int8 else torch.float16,
        no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LLaMADecoderLayer", "LlamaDecoderLayer"]
    )
    print(device_map)
    del model
    return device_map



def load(weight="7",typeof="original"):
    model_id = "/raid/mpsych/LLM/llama/llama2/llama-test/test_weights/{}b".format(weight)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=get_device_map(model_id, "a100-40g", False),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_8bit=False,
        )
    tokenizer = AutoTokenizer.from_pretrained("/raid/mpsych/LLM/llama/llama2/llama-test/test_weights/{}b".format(weight),
                                                      use_fast="/opt" not in model_id)

    generate_kwargs = {
            "max_new_tokens": 400,
            "min_new_tokens": 100,
            "temperature": 0.5,
            "do_sample": False,
            "top_k": 4,
            "penalty_alpha": 0.6,
        }
    return loader(tokenizer, generate_kwargs, model)
    

            
            
class loader:
    def __init__(self, tokenizer,generate_kwargs, model):
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.model = model
        #sself.typeof = typeof
        
    def ask(self,prompt):
         with torch.no_grad():
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            assert len(input_ids) == 1, len(input_ids)
            if input_ids[0][-1] == 2:
                input_ids = input_ids[:, :-1]
            input_ids = input_ids.to(0)
            generated_ids = self.model.generate(
                input_ids,
                **self.generate_kwargs
            )
            result = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
            print(result)
    def finetune(self, typem):
        if typem == "origninal":
            print("using originalmodel")
        if  typem == "custom":
            print("using cutom model")
        else:
            print(typem)
        
        
    
        
        

    
            
            
