import argparse, os, json
import torch
from src.misc import get_model_and_tokenizer


if __name__ == '__main__':

    # define some arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str)
    parser.add_argument("--prompt", type=str, default='prompt.txt')
    parser.add_argument("--output", type=str, default='generate.log')
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    # 载入env_conf
    with open(args.env_conf, 'r') as f:
        env_conf = json.load(f)

    # 载入prompt
    if os.path.exists(args.prompt):
        with open(args.prompt, 'r') as f:
            prompt = f.read()
    else:
        prompt = args.prompt

    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])

    ckp_file = env_conf['model']['save_ckp']
    if os.path.exists(ckp_file):
        print(f"load checkpoint {ckp_file}")
        model.load_checkpoint(ckp_file)
    else:
        print(f"{ckp_file} dose not exists")

    input_ids = torch.tensor(tokenizer(prompt).input_ids, dtype=torch.long)[None,:]

    device = next(iter(model.parameters())).device
    input_ids = input_ids.to(device)

    output = model.generate(
        input_ids=input_ids, 
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
    )[:,input_ids.shape[-1]:]

    if isinstance(output, torch.Tensor):
        output = output.ravel().tolist()

    if tokenizer.eos_token_id in output:
        index = output.index(tokenizer.eos_token_id)
        output = output[:index]

    result = {
        "prompt": input_ids.shape[-1],
        "result": tokenizer.decode(output),
        "length": len(output)
    }
    result = json.dumps(result, indent=4)

    # output the generated result
    with open(args.output, 'w') as f:
        f.write(result)

