import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from tokenmix2.misc import get_model_and_tokenizer

from corpus.processor.conversations import get_conv_template


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name, chat_template):
    if chat_template is not None:
        conv = get_conv_template(chat_template)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

# modified: 将参数max_length去掉
def get_pred(
        env_conf, 
        data, 
        max_gen, 
        prompt_format, 
        dataset, 
        device, 
        model_name, 
        out_path, 
        model_max_length,
        chat_template):
    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
    ckp_file = env_conf['model']['save_ckp']
    if os.path.exists(ckp_file):
        print(f"load checkpoint {ckp_file}")
        model.load_checkpoint(ckp_file)
    else:
        print(f"{ckp_file} dose not exists")

    model.eval()
    
    for json_obj in tqdm(data):

        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        
        # modified: 注释掉
        # tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model_name:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        # if len(tokenized_prompt) > max_length:
        #     half = int(max_length/2)
        #     prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name, chat_template)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        # =================================================================================================
        # NOTE: 注释掉
        # if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
        #     output = model.generate(
        #         **input,
        #         max_new_tokens=max_gen,
        #         num_beams=1,
        #         do_sample=False,
        #         temperature=1.0,
        #         min_length=context_length+1,
        #         eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
        #     )[0]
        # else:
        # =================================================================================================

        # ======================================================
        # NOTE: 供测试使用
        if input.input_ids.shape[-1] >= model_max_length - max_gen:
            input.input_ids = input.input_ids[..., -model_max_length + max_gen:]
        context_length = input.input_ids.shape[-1]
        # ======================================================

    
        # ============================
        # NOTE: 新增加
        if max_gen == 0 and hasattr(model, 'prefill'):
            kv_caches = model.prefill(input_ids=input.input_ids)
            del kv_caches
            output = input.input_ids
        else:
            output = model.generate(
                input_ids=input.input_ids,
                max_new_tokens=max_gen
            ).ravel().tolist()
        # ============================


        # ==============================================
        # NOTE: 新增加
        if tokenizer.eos_token_id in output:
            index = output.index(tokenizer.eos_token_id)
            output = output[:index]
        torch.cuda.empty_cache()
        # ==============================================


        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--chat_template", type=str, default=None)
    parser.add_argument("--max_gen", type=int, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    args = parser.parse_args()

    import json, os
    with open(args.env_conf, "r") as f:
        env_conf = json.load(f)
    with open("test_longbench/pred.json", 'r') as f:
        pred_conf = json.load(f)

    seed_everything(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.env_conf.split('/')[-1]

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    # NOTE: 加载数据集
    for dataset in pred_conf:
        assert dataset in datasets
    datasets = pred_conf

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('LongBench/LongBench.py', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('LongBench/LongBench.py', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        get_pred(
            env_conf, 
            data_all, 
            max_gen if args.max_gen is None else args.max_gen, 
            prompt_format, 
            dataset, 
            device, 
            model_name, 
            out_path, 
            args.model_max_length, 
            args.chat_template)
