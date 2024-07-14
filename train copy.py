from src.misc import get_model_and_tokenizer, get_env_conf, get_data_generator_deepspeed, Saver

import deepspeed, argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_conf', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])
    dataset = get_data_generator_deepspeed(model, tokenizer, **env_conf["train"])
    saver = Saver(model, **env_conf['train'])
    model.train()

    model_engine, optimizer, loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.ft_params(),
        training_data=dataset)

    for step, batch in tqdm(enumerate(loader), disable=args.local_rank != 0):
        batch.update({"local_rank": args.local_rank})
        loss = model_engine(**batch).loss
        model_engine.backward(loss)
        model_engine.step()

        if hasattr(model_engine.model, 'reset'):
            model_engine.model.reset()
        
        if args.local_rank in (0, -1):
            saver.step()