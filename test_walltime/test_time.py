from tokenmix.misc import get_env_conf, get_model_and_tokenizer
import argparse
import torch
import time
from profiler import WallTime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_conf', type=str, default=None)
    args = parser.parse_args()

    lengths = [(x + 1) * 1024 for x in range(2,128)]
    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])
    
    walltime = WallTime("test", cuda=[0,1,2,3,4,5,6,7])
    num_repeats = 5

    for length in lengths:
        fake_input_ids = torch.zeros((1, length), dtype=torch.int64)
        
        for _ in range(num_repeats):
            with torch.no_grad():

                with WallTime.get('test'):
                    if hasattr(model, 'prefill'):
                        model.prefill(input_ids=fake_input_ids)
                    else:
                        model(input_ids=fake_input_ids)

                if hasattr(model.model, 'memory'):
                    model.model.memory.reset()

        walltime.result(postfix=length, detail=True)
        walltime.reset()
        torch.cuda.empty_cache()
