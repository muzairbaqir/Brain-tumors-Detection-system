"""Step 2: Preprocessing and Data Augmentation Check."""
import sys
import logging
import argparse
import torch
from pathlib import Path

# Add thesis_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.data import BrainTumorDataset
from src.visualize import plot_augmentation_effects, plot_tensor_distributions


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main(config_path: str = None):
    logger = setup_logging()
    
    if config_path:
        config = ExperimentConfig.from_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        logger.error("Please provide a config file using --config")
        sys.exit(1)
        
    preprocessing_dir = Path(config.output_dir) / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing preprocessing verification...")
    
    # Load dataset logic
    dataset = BrainTumorDataset(config)
    train_loader, _, _ = dataset.get_dataloaders()
    
    try:
        logger.info("Generating Augmentation Comparison (Original vs. Distorted)...")
        plot_augmentation_effects(dataset, str(preprocessing_dir))
        
        logger.info("Generating Tensor Distributions (Pixel Normalization check)...")
        plot_tensor_distributions(train_loader, str(preprocessing_dir))
        
        logger.info(f"Success! Visualizations saved to {preprocessing_dir}")
        logger.info("Please verify the data looks correct before running Step 3.")
    except Exception as e:
        logger.error(f"Failed to generate preprocessing visuals: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Preprocessing and Augmentation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    main(config_path=args.config)
