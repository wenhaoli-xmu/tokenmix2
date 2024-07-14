import argparse
import torch
import random

from src.misc import get_model_and_tokenizer, get_env_conf, get_optimizer_and_lr_adjuster, get_data_generator
from src.misc import Saver, RMTEvaluator, OptimAnalyzer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default='env_conf.json')
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
    optim, lr_adjuster = get_optimizer_and_lr_adjuster(model, **env_conf["train"])
    data_generator = get_data_generator(model, tokenizer, **env_conf["train"])

    saver = Saver(model, **env_conf["train"])
    evaluator = RMTEvaluator(model, tokenizer, **env_conf["train"])
    analyzer = OptimAnalyzer(model.ft_params())

    accum_grad = env_conf["train"]["accum_grad"]
    clip_grad = env_conf["train"]["clip_grad"]

    torch.random.manual_seed(114514)
    random.seed(114514)

    for iter, (data, io_wrapper) in enumerate(data_generator):

        # adjust the learning rate
        lr_adjuster(step=iter)

        past_memories = []
        past_grads = []

        # forward propagation
        past_memory = None
        for chunk_id, (inputs, compute_loss) in enumerate(io_wrapper.wrap(data)):

            # forward prop
            inputs.update({"memory": past_memory})
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
                memory = model.model.update_memory(**inputs)
                if memory is not None:
                    memory = memory.data
                    memory.requires_grad_(True)
                    assert memory.is_leaf

            # accumulate result
            past_memories += [memory]
            if past_memory is None:
                past_grads += [None]
            elif loss is None:
                past_grads += [torch.zeros_like(past_memory)]
            else:
                past_grads += [past_memory.grad.data.clone()]
            past_memory = past_memories[-1]

            # clear local variables
            del outputs, memory, inputs, compute_loss, loss
            torch.cuda.empty_cache()

        # release memory overhead
        del past_memory
        torch.cuda.empty_cache()

        # backward propagation
        for chunk_id, (inputs, compute_loss) in reversed(tuple(enumerate(io_wrapper.wrap(data)))[:-1]):

            # forward prop & backward prop
            past_memory = past_memories[chunk_id - 1] if chunk_id - 1 >= 0 else None

            inputs.update({"memory": past_memory})
            memory = model.model.update_memory(**inputs)

            if memory.requires_grad:
                memory.backward(gradient=past_grads[chunk_id + 1])

            past_grads[chunk_id] = past_grads[chunk_id] + past_memory.grad.data if chunk_id > 0 else None

            del past_memory, memory, inputs, compute_loss
            torch.cuda.empty_cache()

        # release memory overheads
        del past_grads
        torch.cuda.empty_cache()

        # update the parameters
        if (iter + 1) % accum_grad == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.ft_params(), max_norm=clip_grad)
            analyzer.step()
            optim.step()
            optim.zero_grad()

        # reset the model and clear cuda cache
        model.reset()
        torch.cuda.empty_cache()

        saver.step()
        evaluator.step()
        
