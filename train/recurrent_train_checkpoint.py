import argparse
import torch
import random

from tokenmix2.misc import get_model_and_tokenizer, get_env_conf, get_optimizer_and_lr_adjuster
from tokenmix2.misc import Saver, RMTEvaluator
from tokenmix2.modifier import segment
from corpus import get_processor, LazyRandomSampleCorpus
from torch.utils.data import ConcatDataset, DataLoader
from functools import partial
import wandb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default='env_conf.json')
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
    optim, lr_adjuster = get_optimizer_and_lr_adjuster(model, **env_conf["train"])

    sum_partition = 0
    num_iters = env_conf['train']['train_iters']
    corpus = []
    for info in env_conf['train']['corpus']:
        sum_partition += info['partition']
        num_instance = int(info['partition'] * num_iters)

        proc = get_processor(info['conf'], tokenizer)
        corp = LazyRandomSampleCorpus(info['data'], proc, max_instance=num_instance, use_cache=False)
        corpus.append(corp)

    assert sum_partition == 1
    dataset = ConcatDataset(corpus)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    saver = Saver(model, **env_conf["train"])
    evaluator = RMTEvaluator(model, tokenizer, **env_conf["train"])

    accum_grad = env_conf["train"]["accum_grad"]
    clip_grad = env_conf["train"]["clip_grad"]

    torch.random.manual_seed(42)
    random.seed(42)
    wandb.init(project='gru-transformer')


    def warp_data(input_ids, labels, attention_mask):
        chunk_size = model.conf['chunk_size']
        labels = labels[1:] + [-100]

        input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]
        labels = torch.tensor(labels, dtype=torch.int64)[None, :]
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)[None, :]

        input_ids = segment(input_ids, dim=-1, n=chunk_size)
        labels = segment(labels, dim=-1, n=chunk_size)
        attention_mask = segment(attention_mask, dim=-1, n=chunk_size)

        def compute_loss(outputs, valid_token_num):
            return outputs.loss / valid_token_num

        for inpus, labs in zip(input_ids, labels):
            yield ({
                "input_ids": inpus,
                "labels": labs},
                partial(compute_loss, valid_token_num=(labs != -100).count_nonzero()))


    for step, batch in enumerate(loader):

        # adjust the learning rate
        lr_adjuster(step=step)

        past_memories = []
        past_grads = []
        accum_loss = 0

        # forward propagation
        past_memory = None
        for chunk_id, (inputs, compute_loss) in enumerate(warp_data(**batch)):

            # forward prop
            inputs.update({"memory": past_memory})
            outputs = model(**inputs)

            # backward prop
            if compute_loss is not None:
                loss = compute_loss(outputs) / accum_grad
                accum_loss += loss.item() if loss.isnan().item() is False else 0
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
        for chunk_id, (inputs, compute_loss) in reversed(tuple(enumerate(warp_data(**batch)))[:-1]):

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
        if (step + 1) % accum_grad == 0:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.ft_params(), max_norm=clip_grad)
            optim.step()
            optim.zero_grad()

        # reset the model and clear cuda cache
        model.reset()
        torch.cuda.empty_cache()

        saver.step()
        print(accum_loss)
        wandb.log({"Train": {"Samples": {"train_loss": accum_loss}}})

    wandb.finish()
