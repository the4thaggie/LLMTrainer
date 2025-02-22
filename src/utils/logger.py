import logging
from torch.utils.tensorboard import SummaryWriter
import wandb

class UnifiedLogger:
    """
    Combines multiple logging channels:
    - Console (color-coded levels)
    - File (JSON format)
    - WandB (experiment tracking)
    - TensorBoard (performance metrics)
    """
    def __init__(self, log_file=None, tb_log_dir=None, wandb_project=None):
        # Set up the console logger.
        self.logger = logging.getLogger("UnifiedLogger")
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Optionally set up file logging.
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Optionally initialize WandB.
        if wandb_project:
            wandb.init(project=wandb_project)
            self.wandb = wandb
        else:
            self.wandb = None

        # Optionally initialize TensorBoard.
        if tb_log_dir:
            self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        else:
            self.tb_writer = None

    def log_step(self, step, loss, lr):
        # Log metrics to WandB and TensorBoard if available.
        if self.wandb:
            self.wandb.log({"loss": loss, "lr": lr}, step=step)
        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss", loss, step)
        # Also log to the console/file.
        self.logger.info(f"Step: {step}, Loss: {loss}, Learning Rate: {lr}")

def setup_logger(log_file=None, tb_log_dir=None, wandb_project=None):
    """
    Initializes and returns a UnifiedLogger instance.
    
    :param log_file: Optional filepath for file logging.
    :param tb_log_dir: Optional directory for TensorBoard logs.
    :param wandb_project: Optional WandB project name.
    :return: An instance of UnifiedLogger.
    """
    return UnifiedLogger(log_file=log_file, tb_log_dir=tb_log_dir, wandb_project=wandb_project)
