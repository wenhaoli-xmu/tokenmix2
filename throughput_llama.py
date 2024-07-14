from src.misc import get_corpus
from src.io_wrapper import SegmentRecurrentIOWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

import time, pynvml
import torch


if __name__ == '__main__':
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token, device_map='auto', torch_dtype=torch.bfloat16)

    corpus = get_corpus("pg19.test.1m")
    data = corpus[0]
    data['text'] = data['text'] + data['text']
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)   

    times = []
    memms = []

    try:
        for truncation in range(0, 99328, 1024):

            io_wrapper = SegmentRecurrentIOWrapper(
                tokenizer, 
                chunk_size=99328, 
                truncation=truncation)
            
            start_time = time.time()
            
            with torch.no_grad():
                for chunk_inputs, _ in io_wrapper.wrap(data):
                    model(chunk_inputs['input_ids'])

            end_time = time.time()
            inc_time = end_time - start_time
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            cur_mems = memory_info.used

            print(inc_time)
            print(cur_mems)

            times.append((truncation, inc_time))
            memms.append((truncation, cur_mems))
    except:
        pass

    torch.save([times, memms], "llama-4k.pt")
    import IPython
    IPython.embed(header='debug')            


