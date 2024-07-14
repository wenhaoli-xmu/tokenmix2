from src.misc import get_model_and_tokenizer, get_env_conf, get_corpus
from src.io_wrapper import SegmentRecurrentIOWrapper
import argparse

import time, pynvml
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_conf', default=None)
    args = parser.parse_args()

    
    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
    corpus = get_corpus("pg19.test.1m")
    data = corpus[0]
    data['text'] = data['text'] + data['text']
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)   

    times = []
    memms = []

    for truncation in range(0, 99328, 1024):

        io_wrapper = SegmentRecurrentIOWrapper(
            tokenizer, 
            chunk_size=model.chunk_size, 
            truncation=truncation)
        
        start_time = time.time()
        
        with torch.no_grad():
            for chunk_inputs, _ in io_wrapper.wrap(data):
                chunk_inputs.update({"do_not_decode": True})
                model(**chunk_inputs)

        end_time = time.time()
        inc_time = end_time - start_time
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        cur_mems = memory_info.used

        print(inc_time)
        print(cur_mems)

        times.append((truncation, inc_time))
        memms.append((truncation, cur_mems))

        model.reset()

    torch.save([times, memms], "uio-record-100k.pt")
    import IPython
    IPython.embed(header='debug')            


