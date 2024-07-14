from src.modifier import Modifier
from peft import LoraConfig, get_peft_model, TaskType


class TinyLlama(Modifier):
    def _init_lora(
            self,
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float):

        target_modules = r".*\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        self.decoder = get_peft_model(self.model, peft_config)


    def __init__(self, model, save_ckp, load_ckp, config):
        super().__init__(model, save_ckp, load_ckp)
        self._init_lora(lora_rank=128, lora_alpha=512, lora_dropout=0)
        self.model.gradient_checkpointing_enable()
        self.model.train()


    def ft_params(self):
        params = []

        for layer in self.model.base_model.layers:
            params += [
                layer.self_attn.q_proj.lora_A.default.weight,
                layer.self_attn.q_proj.lora_B.default.weight,
                layer.self_attn.k_proj.lora_A.default.weight,
                layer.self_attn.k_proj.lora_B.default.weight,
                layer.self_attn.v_proj.lora_A.default.weight,
                layer.self_attn.v_proj.lora_B.default.weight,
                layer.self_attn.o_proj.lora_A.default.weight,
                layer.self_attn.o_proj.lora_B.default.weight,
                layer.mlp.gate_proj.lora_A.default.weight,
                layer.mlp.gate_proj.lora_B.default.weight,
                layer.mlp.up_proj.lora_A.default.weight,
                layer.mlp.up_proj.lora_B.default.weight,
                layer.mlp.down_proj.lora_A.default.weight,
                layer.mlp.down_proj.lora_B.default.weight]
            
        return params
        
    
    def reset(self):
        pass


    def forward(self, input_ids, labels=None, **kwargs):
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0, 1)
        if labels is not None and labels.ndim == 3:
            labels = labels.flatten(0, 1)

        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        return self.model(input_ids=input_ids, labels=labels)
