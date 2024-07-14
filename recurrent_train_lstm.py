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

        past_cells = []
        past_state = []
        state_grad = []
        cells_grad = []

        # forward propagation
        last_cells = None
        last_state = None
        for chunk_id, (inputs, compute_loss) in enumerate(io_wrapper.wrap(data)):

            # forward prop
            inputs.update({"state": last_state, "use_mem": False})
            outputs = model(**inputs)

            # backward prop
            if compute_loss is not None:
                loss = compute_loss(outputs) / accum_grad
                print(f"loss: {loss.item()}", flush=True)
                if loss.requires_grad:
                    loss.backward()
            else:
                loss = None

            # update memory
            with torch.no_grad():
                inputs.update({"cells": last_cells})
                cells, state = model.model.update_memory(**inputs)
                if cells is not None:
                    cells = cells.data
                    state = state.data
                    cells.requires_grad_(True)
                    state.requires_grad_(True)
                    assert cells.is_leaf and state.is_leaf

            # accumulate result
            past_cells += [cells]
            past_state += [state]
            if last_cells is None:
                cells_grad += [None]
                state_grad += [None]
            elif loss is None:
                cells_grad += [torch.zeros_like(last_cells)]
                state_grad += [torch.zeros_like(last_state)]
            else:
                cells_grad += [torch.zeros_like(last_cells)]
                state_grad += [last_state.grad.data.clone()]
            last_cells = past_cells[-1]
            last_state = past_state[-1]

            # clear local variables
            del outputs, cells, state, inputs, compute_loss, loss
            torch.cuda.empty_cache()

        # release memory overhead
        del last_cells, last_state
        torch.cuda.empty_cache()

        # backward propagation
        for chunk_id, (inputs, compute_loss) in reversed(tuple(enumerate(io_wrapper.wrap(data)))[:-1]):

            # forward prop & backward prop
            last_cells, last_state = past_cells[chunk_id - 1], past_state[chunk_id - 1] if chunk_id - 1 >= 0 else None
            inputs.update({"cells": last_cells, "state": last_state, "use_mem": False})
            cells, state = model.model.update_memory(**inputs)

            memory = torch.cat([cells, state], dim=0)
            memory.backward(gradient=torch.cat([cells_grad[chunk_id + 1], state_grad[chunk_id + 1]], dim=0))

            cells_grad[chunk_id] = cells_grad[chunk_id] + last_cells.grad.data if chunk_id > 0 else None
            state_grad[chunk_id] = state_grad[chunk_id] + last_state.grad.data if chunk_id > 0 else None

            del last_cells, last_state, cells, state, memory, inputs, compute_loss
            torch.cuda.empty_cache()

        # release memory overheads
        del cells_grad, state_grad
        torch.cuda.empty_cache()

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
        evaluator.step_rmt()
