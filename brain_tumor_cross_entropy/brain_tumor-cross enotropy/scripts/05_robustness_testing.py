"""Step 5: Robustness Evaluation on Corrupted MRI Scans."""
import sys
import logging
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Add thesis_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.data import BrainTumorDataset
from src.models import ModelFactory
from src.robustness import MedicalCorruptions

def setup_logging(log_dir: Path, experiment_name: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_name}_robustness.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)

def evaluate_corruption(model, dataloader, corruption_name, severity, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Apply corruption
            corrupted_images = []
            for img in images:
                corrupted = MedicalCorruptions.apply_corruption(img, corruption_name, severity)
                corrupted_images.append(corrupted)
            
            corrupted_batch = torch.stack(corrupted_images).to(device)
            labels = labels.to(device)
            
            outputs = model(corrupted_batch)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds)

def plot_robustness(results, output_dir):
    plt.figure(figsize=(10, 6))
    
    for corruption, scores in results.items():
        severities = list(scores.keys())
        accuracies = list(scores.values())
        plt.plot(severities, accuracies, marker='o', linewidth=2, label=corruption.replace('_', ' ').title())
        
    plt.title("Model Robustness Under Clinical Corruptions", fontsize=14, pad=15)
    plt.xlabel("Corruption Severity Level", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = Path(output_dir) / "robustness_degradation_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def main(config_path: str = None):
    if not config_path:
        print("Please provide a config file using --config")
        sys.exit(1)
        
    config = ExperimentConfig.from_yaml(config_path)
    log_dir = Path(config.log_dir)
    logger = setup_logging(log_dir, config.experiment_name)
    logger.info("--- Starting Phase 5: Robustness Testing ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Loading Test Dataset...")
    dataset = BrainTumorDataset(config)
    _, _, test_loader = dataset.get_dataloaders()
    
    logger.info(f"Initializing model: {config.model.name}")
    model = ModelFactory.create_model(config).to(device)
    
    checkpoint_path = Path(config.checkpoint_dir) / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(f"Cannot find trained model checkpoint at {checkpoint_path}!")
        sys.exit(1)
        
    logger.info("Loading trained weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    corruptions = config.raw_config.get('robustness', {}).get('corruptions', [])
    severities = config.raw_config.get('robustness', {}).get('severity_levels', [1, 2, 3, 4, 5])
    
    if not corruptions:
        logger.error("No corruptions found in config file!")
        sys.exit(1)
        
    results = {c: {} for c in corruptions}
    
    for corruption in corruptions:
        logger.info(f"Testing Corruption: {corruption}")
        for severity in severities:
            acc = evaluate_corruption(model, test_loader, corruption, severity, device)
            results[corruption][severity] = acc
            logger.info(f"  Severity {severity}: Accuracy = {acc:.4f}")
            
    logger.info("Generating Robustness Plot...")
    plot_path = plot_robustness(results, config.output_dir)
    logger.info(f"Saved plot to {plot_path}")
    logger.info("Robustness Testing Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(config_path=args.config)
