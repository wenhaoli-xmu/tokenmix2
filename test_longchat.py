from src.misc import get_env_conf
from src.misc import Evaluator

DEVICE_MAP = {
    "model.beacon_embed_tokens": 0,
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 0,
    "model.layers.4": 1,
    "model.layers.5": 1,
    "model.layers.6": 1,
    "model.layers.7": 1,
    "model.layers.8": 2,
    "model.layers.9": 2,
    "model.layers.10": 2,
    "model.layers.11": 2,
    "model.layers.12": 3,
    "model.layers.13": 3,
    "model.layers.14": 3,
    "model.layers.15": 3,
    "model.layers.16": 4,
    "model.layers.17": 4,
    "model.layers.18": 4,
    "model.layers.19": 4,
    "model.layers.20": 5,
    "model.layers.21": 5,
    "model.layers.22": 5,
    "model.layers.23": 5,
    "model.layers.24": 6,
    "model.layers.25": 6,
    "model.layers.26": 6,
    "model.layers.27": 6,
    "model.layers.28": 7,
    "model.layers.29": 7,
    "model.layers.30": 7,
    "model.layers.31": 7,
    "model.norm": 7,
    "lm_head": 7
}


def get_model_and_tokenizer():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "Yukang/Llama-2-7b-longlora-32k-ft"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token="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")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map=DEVICE_MAP, token="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")
    model.eval()

    return tokenizer, model


if __name__ == '__main__':
    test_conf = get_env_conf("test.json")

    tokenizer, model = get_model_and_tokenizer()
    evaluator = Evaluator(model, tokenizer, eval=None, tasks=test_conf)
    evaluator.evaluate()
