class Config:
    def __init__(self):
        self.base_model = "unsloth/DeepSeek-R1-Distill-Llama-8B"
        self.max_seq_length = 2048
        self.quantization = "4bit"
        self.lora_rank = 16
        self.target_modules = ["q_proj", "v_proj"]
        self.checkpoint_dir = "checkpoints"
        self.vector_db_url = "http://localhost:6333"
