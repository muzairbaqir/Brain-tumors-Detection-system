import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    dataset_path: str  # Path to Kaggle dataset
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    
    # Augmentation
    enable_augmentation: bool = True
    aug_rotation: int = 10  # Conservative for medical images
    aug_brightness: float = 0.1
    aug_contrast: float = 0.1
    aug_horizontal_flip: bool = False  # Be careful with medical images


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "swin_tiny_patch4_window7_224"  # timm model name
    pretrained: bool = True
    num_classes: int = 4
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    epochs: int = 30
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # cosine, linear, step
    
    # Loss
    use_class_weights: bool = True
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1
    
    # Early stopping
    patience: int = 5
    val_check_interval: int = 1  # Check every N epochs
    
    # Checkpointing
    save_best_only: bool = True
    save_last: bool = True


@dataclass
class RobustnessConfig:
    """Robustness evaluation configuration."""
    # Corruption types to test
    corruptions: List[str] = None  # gaussian_noise, motion_blur, elastic_transform, etc.
    severity_levels: List[int] = None  # [1, 2, 3, 4, 5]
    
    def __post_init__(self):
        if self.corruptions is None:
            self.corruptions = [
                "gaussian_noise",
                "motion_blur",
                "elastic_deform",
                "intensity_shift",
                "spike_noise",
            ]
        if self.severity_levels is None:
            self.severity_levels = [1, 2, 3, 4, 5]


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Metadata (required fields first)
    experiment_name: str
    
    # Sub-configs (required)
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    robustness: RobustnessConfig
    
    # Optional fields with defaults
    description: str = ""
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/model_results/checkpoints"
    log_dir: str = "./outputs/model_results/logs"
    mode: str = "train"  # train, eval_robustness, train_robust
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        data_cfg = DataConfig(**cfg_dict.get('data', {}))
        model_cfg = ModelConfig(**cfg_dict.get('model', {}))
        training_cfg = TrainingConfig(**cfg_dict.get('training', {}))
        robustness_cfg = RobustnessConfig(**cfg_dict.get('robustness', {}))
        
        return cls(
            experiment_name=cfg_dict['experiment_name'],
            description=cfg_dict.get('description', ''),
            data=data_cfg,
            model=model_cfg,
            training=training_cfg,
            robustness=robustness_cfg,
            output_dir=cfg_dict.get('output_dir', './outputs'),
            checkpoint_dir=cfg_dict.get('checkpoint_dir', f"{cfg_dict.get('output_dir', './outputs')}/model_results/checkpoints"),
            log_dir=cfg_dict.get('log_dir', f"{cfg_dict.get('output_dir', './outputs')}/model_results/logs"),
            mode=cfg_dict.get('mode', 'train'),
        )
    
    def to_yaml(self, output_path: str):
        """Save config to YAML."""
        cfg_dict = {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'robustness': asdict(self.robustness),
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'mode': self.mode,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)


"""
Kaggle Direct Dataset Loader
Loads Brain Tumor MRI dataset directly from Kaggle without local storage
"""
