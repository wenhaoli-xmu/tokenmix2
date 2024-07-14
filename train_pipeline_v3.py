from src.misc import get_model_and_tokenizer_model_args, get_env_conf, get_data_generator_deepspeed, Saver, Evaluator, get_optimizer_and_lr_adjuster

import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torch
import wandb
from corpus import DataProcessor, Corpus, json_to_dataconfig
import os, json
import concurrent

from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments, HfArgumentParser
from typing import Optional, List


@dataclass
class ModelArguments:
    model_name: str = field(default="togethercomputer/Llama-2-7B-32K")
    model_dtype: str = field(default='bf16')
    model_version: str = field(default='hybird9')
    model_structure: str = field(default='llama')
    device_map: str = field(default='auto')
    config: str = field(default=None)


@dataclass
class DataArguments:
    train_data: Optional[List[str]] = field(default=None)



if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = get_model_and_tokenizer_model_args(model_args)
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_conf', type=str, default=None)
    args = parser.parse_args()
    
    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(model_args)
    optim, lr_adjuster = get_optimizer_and_lr_adjuster(model, **env_conf['train'])
    # dataset = get_data_generator_deepspeed(model, tokenizer, **env_conf["train"])
    saver = Saver(model, **env_conf['train'])
    evaluator = Evaluator(model, tokenizer, **env_conf['train'])
    model.train()


    # load datasets
    sum_partition = 0
    num_iters = env_conf["train"]["train_iters"]
    corpus = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for info in env_conf["train"]["corpus"]:
            conf_path = info["conf"]
            data_path = info["data"]
            partition = info["partition"]
            sum_partition += partition

            num_instance = int(partition * num_iters)

            with open(conf_path, 'r') as f:
                conf = json_to_dataconfig(json.load(f))
            processor = DataProcessor(conf, tokenizer)

            futures.append(executor.submit(Corpus, data_path, processor, max_instance=num_instance, random_sample=False))

        for future in concurrent.futures.as_completed(futures):
            corpus.append(future.result())

    dataset = ConcatDataset(corpus)


    wandb.init(project='hybird')

    accum_grad = env_conf["train"]["accum_grad"]
    clip_grad = env_conf["train"]["clip_grad"]

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for step, batch in tqdm(enumerate(loader), disable=True):
        lr_adjuster(step=step)
        loss = model(**batch).loss / accum_grad
        loss.backward()

        wandb.log({
            "Train": {
                "Samples": {
                    "train_loss": loss.item()
                }
            }
        })

        # update the parameters
        if (step + 1) % accum_grad == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.ft_params(), max_norm=clip_grad)
            optim.step()
            optim.zero_grad()

        # reset the model and clear cuda cache
        model.reset()
        saver.step()
        evaluator.step()

        torch.cuda.empty_cache()

    wandb.finish()
