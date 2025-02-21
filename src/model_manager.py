class ModelLoader:
    """
    Handles 4-bit model loading with LoRA adapters
    Features:
    - Automatic device mapping
    - Quantization config
    - Layer freezing
    """
    def apply_optimizations(self, model):
        model = prepare_model_for_kbit_training(model)
        return get_peft_model(model, LoraConfig(
            r=self.config.lora_rank,
            target_modules=self.config.target_modules
        ))
