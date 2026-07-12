"""Step 3: Model Architecture and Training Loop."""
import sys
import logging
import argparse
import torch
import numpy as np
from pathlib import Path

# Add thesis_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.data import BrainTumorDataset
from src.models import ModelFactory, create_optimizer, create_scheduler, create_loss
from src.training import Trainer
from src.evaluation import plot_training_history


def setup_logging(log_dir: Path, experiment_name: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_name}_training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def main(config_path: str = None):
    if not config_path:
        print("Please provide a config file using --config")
        sys.exit(1)
        
    config = ExperimentConfig.from_yaml(config_path)
    log_dir = Path(config.log_dir)
    logger = setup_logging(log_dir, config.experiment_name)
    logger.info(f"Loaded config from {config_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(config.data.seed)
    np.random.seed(config.data.seed)
    
    logger.info("Loading dataset...")
    dataset = BrainTumorDataset(config)
    train_loader, val_loader, _ = dataset.get_dataloaders()
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    logger.info(f"Creating model: {config.model.name}")
    model = ModelFactory.create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    class_weights = torch.ones(dataset.num_classes).to(device)
    loss_fn = create_loss(config, class_weights=class_weights)
    
    logger.info("Starting training loop...")
    trainer = Trainer(config, model, optimizer, scheduler, loss_fn, device)
    trainer.train(train_loader, val_loader)
    
    logger.info("Training complete! Generating training history visualization...")
    plot_training_history(trainer.history, Path(config.output_dir) / "model_results")
    logger.info("Saved training history plot. You can now run Step 4 (Testing).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    main(config_path=args.config)
