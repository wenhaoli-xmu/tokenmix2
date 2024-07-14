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


def destroy_graph(model, eliminate_id):
    try:
        grads, states = model.get_memories(eliminate_id)
        states.backward(gradient=grads)
    except Exception:
        # 有时候这个节点没有产生memory
        # 也有时候这些产生的memory没有被用到
        pass


def remove(reservoir, chunk_id):
    if chunk_id in reservoir:
        reservoir.remove(chunk_id)
    return reservoir


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
    tbptt_window = env_conf["train"]["tbptt"]

    torch.random.manual_seed(0)
    random.seed(0)

    for iter, (data, io_wrapper) in enumerate(data_generator):

        # adjust the learning rate
        lr_adjuster(step=iter)

        # reservoir sampling
        reservoir = []

        # randomized incremental TBPTT
        for chunk_id, (inputs, compute_loss) in enumerate(io_wrapper.wrap(data)):
            outputs = model(**inputs)

            if compute_loss is not None:
                loss = compute_loss(outputs) / accum_grad

                if loss.requires_grad is True:
                    loss.backward(retain_graph=True)
                    print(f"loss: {loss.item()}", flush=True)

            for seg_id in reversed(range(max(0, chunk_id - tbptt_window), chunk_id)):
                retain_graph = False if seg_id == chunk_id - tbptt_window else True
                grads, states = model.get_memories(seg_id)
                states.backward(grads, retain_graph=retain_graph)

        # update the parameters
        if (iter + 1) % accum_grad == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.ft_params(), max_norm=clip_grad)
            optim.step()
            optim.zero_grad()

        # reset & free cache
        model.reset()
        torch.cuda.empty_cache()

        saver.step()
        evaluator.step()
