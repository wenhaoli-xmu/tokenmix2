from src.misc import get_env_conf
from src.misc import Evaluator


def get_model_and_tokenizer():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token="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")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto", token="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")
    model.eval()

    return tokenizer, model


if __name__ == '__main__':
    test_conf = get_env_conf("test.json")

    tokenizer, model = get_model_and_tokenizer()
    evaluator = Evaluator(model, tokenizer, eval=None, tasks=test_conf)
    evaluator.evaluate()
