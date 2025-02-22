from unsloth import FastLanguageModel
from peft import LoraConfig

class ModelLoader:
    def __init__(self, config):
        self.config = config

    def load_base_model(self):
        return FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.quantization == "4bit"
        )

    def apply_lora(self, model):
        return FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_rank,
            target_modules=self.config.target_modules,
            lora_dropout=0.1
        )

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, model_id, model_path):
        self.models[model_id] = model_path

    def get_model_path(self, model_id):
        return self.models.get(model_id)

    def predict(self, model_id, input_text):
        # Implementation depends on how you want to handle inference
        pass
