import argparse
import torch
import random

from src.misc import get_model_and_tokenizer, get_env_conf, get_optimizer_and_lr_adjuster, get_data_generator
from src.misc import Saver, Evaluator


def segment(tensor, dim, n):
    total_length = tensor.shape[dim]

    for start in range(0, total_length, n):
        end = min(start + n, total_length)
        indices = [slice(None)] * tensor.ndim
        indices[dim] = slice(start, end)
        yield tensor[tuple(indices)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default='env_conf.json')
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
    optim, lr_adjuster = get_optimizer_and_lr_adjuster(model, **env_conf["train"])
    data_generator = get_data_generator(model, tokenizer, **env_conf["train"])
    saver = Saver(model, **env_conf["train"])
    evaluator = Evaluator(model, tokenizer, **env_conf["train"])

    accum_grad = env_conf["train"]["accum_grad"]
    clip_grad = env_conf["train"]["clip_grad"]

    torch.random.manual_seed(0)
    random.seed(0)

    for iter, (data, io_wrapper) in enumerate(data_generator):

        # adjust the learning rate
        lr_adjuster(step=iter)

        past_memories = []
        past_grads = []

        # forward propagation
        total_loss = torch.tensor([0], dtype=torch.float32)
        memory = None

        for chunk_id, (inputs, compute_loss) in enumerate(io_wrapper.wrap(data)):
            inputs.update({"memory": memory})

            outputs = model(**inputs)
            loss = compute_loss(outputs) / accum_grad
            total_loss += loss

            memory = model.model.update_memory(**inputs)

        total_loss.backward()

        # update the parameters
        if (iter + 1) % accum_grad == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.ft_params(), max_norm=clip_grad)
            optim.step()
            optim.zero_grad()

        # reset the model and clear cuda cache
        model.reset()
        torch.cuda.empty_cache()

        saver.step()
        evaluator.step()
