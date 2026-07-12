"""Step 1: Dataloading and Raw Data Verification."""
import sys
import logging
import argparse
from pathlib import Path

# Add thesis_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.visualize import plot_class_distribution, plot_raw_samples


def setup_logging():
    """Setup console logging."""
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
        
    raw_data_dir = Path(config.output_dir) / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing raw dataloading verification...")
    
    # Strip kaggle:// prefix if present
    dataset_path = config.data.dataset_path
    if dataset_path.startswith("kaggle://"):
        dataset_path = dataset_path.replace("kaggle://", "")
        
    try:
        logger.info("Generating Class Distribution chart...")
        plot_class_distribution(dataset_path, str(raw_data_dir))
        
        logger.info("Generating Raw Samples grid...")
        plot_raw_samples(dataset_path, str(raw_data_dir))
        
        logger.info(f"Success! Visualizations saved to {raw_data_dir}")
        logger.info("Please inspect the images to verify data integrity before moving to Step 2.")
    except Exception as e:
        logger.error(f"Failed to generate dataloading visuals: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Dataloading and Raw Data Check")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    main(config_path=args.config)
