class TrainingCoordinator:
    """
    Manages end-to-end training process with:
    - Automatic resume functionality
    - Mixed precision training
    - Gradient checkpointing
    """
    def create_progress_bars(self):
        self.main_bar = tqdm(total=self.total_steps, desc="Training")
        self.mem_bar = tqdm(bar_format="{desc}", position=1)
        
    def update_memory_display(self):
        self.mem_bar.set_description(
            f"VRAM: {self.monitor.free_mem_gb()}GB free | "
            f"Batch Size: {self.current_batch_size}"
        )
