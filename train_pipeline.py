from tokenmix.misc import get_model_and_tokenizer, get_env_conf, get_data_generator_deepspeed, Saver, Evaluator, get_optimizer_and_lr_adjuster

import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_conf', type=str, default=None)
    args = parser.parse_args()
    
    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])
    optim, lr_adjuster = get_optimizer_and_lr_adjuster(model, **env_conf['train'])
    dataset = get_data_generator_deepspeed(model, tokenizer, **env_conf["train"])
    saver = Saver(model, **env_conf['train'])
    evaluator = Evaluator(model, tokenizer, **env_conf['train'])
    model.train()

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
