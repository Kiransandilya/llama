from llama import Llama
#import torch
import socket as s
import random
import os
import time
import llamarun
#from fairscale.nn.model_parallel.initialize import initialize_model_parallel
class W07:
    def __init__(self):
        ckpt_dir = "/raid/mpsych/LLM/llama/llama2/weights/llama-2-7b-chat"
        tokenizer_path = "/raid/mpsych/LLM/llama/llama2/weights/tokenizer.model"
        temperature = 0.6
        top_p = 0.9
        max_seq_len = 512
        max_gen_len = None
        max_batch_size = 4
        #local_rank = int(os.environ.get("LOCAL_RANK",-1))
        #world_size = int(os.environ.get("WORLD_SIZE",-1))
        os.environ['RANK'] ='0'
        os.environ['WORLD_SIZE'] = '1'
        #os.environ['OMP_NUM_THREADDS'] = '2'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self.port_generator())
        #torch.distributed.init_process_group(backend="nccl",rank=0,world_size=2)
       
        
        #initialize_model_parallel(2)
        #torch.cuda.set_device(local_rank)
        #torch.manual_seed(1)
        

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=1
        )

    def port_generator(self):
        port_number = random.randint(10000, 60000)
        return port_number

    def ask(self, user_input):
        
        print("Generating Output, Please wait......")
        start = time.time()
        user_input_formatted = [
        [{"role": "user", "content": str(user_input)}],  ]
        results = self.generator.chat_completion(
            user_input_formatted,
            max_gen_len=None,
            temperature=0.6,
            top_p=0.9
        )
        end = time.time()
        time_taken = end-start
        print("Generated output in {} seconds".format(time_taken))
        for dialog, result in zip(user_input_formatted, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")



