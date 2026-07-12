"""Step 4: Testing and Advanced Analytics (t-SNE, Saliency, etc)."""
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
from src.models import ModelFactory
from src.evaluation import Evaluator


def setup_logging(log_dir: Path, experiment_name: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_name}_testing.log"
    
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
    
    logger.info("Loading dataset for testing...")
    dataset = BrainTumorDataset(config)
    _, _, test_loader = dataset.get_dataloaders()
    logger.info(f"Test size: {len(test_loader.dataset)}")
    
    logger.info(f"Initializing empty model: {config.model.name}")
    model = ModelFactory.create_model(config).to(device)
    
    checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(f"Cannot find trained model checkpoint at {checkpoint_path}!")
        logger.error("Please run scripts/03_model_training.py first.")
        sys.exit(1)
        
    logger.info(f"Loading trained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Ensure compatibility with both bare state_dicts and our trainer dicts
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    logger.info("Weights loaded successfully.")
    
    logger.info("Running advanced evaluation on unseen test data...")
    evaluator = Evaluator(config)
    test_results = evaluator.evaluate(
        model,
        test_loader,
        dataset.class_names,
        device,
    )
    
    logger.info("Testing complete! Analytics generated.")
    logger.info(f"Check {config.output_dir}/model_results for all your final graphs!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Testing and Explainable AI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    main(config_path=args.config)
