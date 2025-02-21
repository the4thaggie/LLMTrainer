class UnifiedLogger:
    """
    Combines multiple logging channels:
    - Console (color-coded levels)
    - File (JSON format)
    - WandB (experiment tracking)
    - TensorBoard (performance metrics)
    """
    def log_step(self, step, loss, lr):
        self.wandb.log({"loss": loss, "lr": lr}, step=step)
        self.tb_writer.add_scalar("train/loss", loss, step)
