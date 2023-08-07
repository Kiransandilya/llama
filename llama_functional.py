#inditalise the model
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
from typing import Optional
import time
#import fire

from llama import Llama


ckpt_dir='/raid/mpsych/LLM/llama/llama2/weights/llama-2-7b/',
tokenizer_path='/raid/mpsych/LLM/llama/llama2/weights/tokenizer.model',
temperature: float = 0.6,
top_p: float = 0.9,
max_seq_len: int = 512,
max_batch_size: int = 4,
max_gen_len: Optional[int] = None,

#setting the rank and world, to be updated to parallel processing

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '11234'

print("Initialising model, Please wait........")

start = time.time()
generator = Llama.build(
    ckpt_dir='/raid/mpsych/LLM/llama/llama2/weights/llama-2-7b-chat/',
    tokenizer_path='/raid/mpsych/LLM/llama/llama2/weights/tokenizer.model',
    max_seq_len=512,
    max_batch_size=4,
)
end = time.time()
tot_time = end-start
print("Model Initialised in {} seconds".format(tot_time))






def ask(user_input):
    dialogs = [
    [{"role": "user", "content": "{}".format(user_input)}],  ]
    results = generator.chat_completion(
    dialogs,  # type: ignore
    max_gen_len=None,
    temperature=0.6,
    top_p=0.9,)
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print("##########")
        print(
            f">{result['generation']['content']}"
        )
        print("\n==================================\n")
    
while True:
    user_input = input("Please enter your prompt: ")
    user_input_formatted = [
    [{"role": "user", "content": user_input}],  ]
    ask(user_input_formatted)




















#set the paths



#take the input from the user




#produce output