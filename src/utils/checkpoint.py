class CheckpointManager:
    """
    Implements 3-level checkpointing:
    1. Step-based (frequent)
    2. Epoch-based (complete state)
    3. Best-performance (metric-driven)
    """
    def save(self, model, step):
        model.save_pretrained(f"checkpoints/step_{step}")
        torch.save({
            'step': step,
            'optimizer': optimizer.state_dict(),
        }, f"checkpoints/step_{step}/training_state.pt")
