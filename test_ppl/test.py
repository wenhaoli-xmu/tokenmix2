from tokenmix2.misc import get_model_and_tokenizer
from tokenmix2.misc import get_env_conf
from tokenmix2.misc import Evaluator, RMTEvaluator, ENCEvaluator
import argparse, os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--use_env_conf_tasks", action="store_true", default=False)
    parser.add_argument('--rmt', action='store_true', default=False)
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    test_conf = get_env_conf("test_ppl/test.json")

    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
    model.eval()

    ckp_file = env_conf['model']['save_ckp']
    if os.path.exists(ckp_file):
        print(f"load checkpoint {ckp_file}")
        model.load_checkpoint(ckp_file)
    else:
        print(f"{ckp_file} dose not exists")

    if args.rmt:
        evaluator_class = RMTEvaluator
    else:
        evaluator_class = Evaluator

    if args.use_env_conf_tasks:
        evaluator = evaluator_class(model, tokenizer, **env_conf["train"])
    else:
        evaluator = evaluator_class(model, tokenizer, eval=None, tasks=test_conf)
    
    evaluator.evaluate()
