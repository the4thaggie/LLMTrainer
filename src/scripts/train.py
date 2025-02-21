def main():
    config = load_config()
    loader = ModelLoader(config)
    engine = TrainingEngine(config)
    
    try:
        engine.train()
    except KeyboardInterrupt:
        engine.save_checkpoint("interrupted")
        logger.warning("Training paused - safe to exit")
