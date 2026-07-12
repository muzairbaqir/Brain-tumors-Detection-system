from src.config import ExperimentConfig
from src.kaggle_loader import KaggleDirectDataset

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, datasets


class BrainTumorDataset:
    """Handles data loading, splitting, and balancing for brain tumor dataset."""
    
    def __init__(self, config):
        self.config = config
        self.is_kaggle = config.data.dataset_path.startswith("kaggle://")
        
        if self.is_kaggle:
            from src.kaggle_loader import KaggleDirectLoader
            self.kaggle_loader = KaggleDirectLoader(cache_dir="./kaggle_cache", use_cache=True)
            self.class_names = self.kaggle_loader.CLASS_NAMES
            self.num_classes = len(self.class_names)
            self.batch_size = config.data.batch_size
            self.num_workers = 0  # Multiprocessing with Kaggle loader causes issues with API
            self.seed = config.data.seed
        else:
            self.dataset_path = Path(config.data.dataset_path)
            self.img_size = config.data.img_size
            self.batch_size = config.data.batch_size
            self.num_workers = config.data.num_workers
            self.seed = config.data.seed
            
            self._validate_paths()
            self.class_names = self._get_class_names()
            self.num_classes = len(self.class_names)
        
    def _validate_paths(self):
        """Verify dataset exists."""
        train_dir = self.dataset_path / "Training"
        test_dir = self.dataset_path / "Testing"
        
        assert train_dir.exists(), f"Training path not found: {train_dir}"
        assert test_dir.exists(), f"Testing path not found: {test_dir}"
    
    def _get_class_names(self) -> list:
        """Extract class names from directory structure."""
        train_dir = self.dataset_path / "Training"
        classes = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(train_dir / d)])
        return classes
    
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Return train and test transforms."""
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        
        # Train transforms (with augmentation)
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomRotation(self.config.data.aug_rotation),
            transforms.ColorJitter(
                brightness=self.config.data.aug_brightness,
                contrast=self.config.data.aug_contrast
            ),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        
        # Test transforms (no augmentation)
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
        
        return train_transform, test_transform
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test splits with proper balancing.
        
        Returns:
            train_loader, val_loader, test_loader
        """
        if self.is_kaggle:
            return self._get_kaggle_dataloaders()
            
        train_transform, test_transform = self.get_transforms()
        
        # Load full training dataset with augmentation
        train_dir = self.dataset_path / "Training"
        full_train_dataset = datasets.ImageFolder(
            str(train_dir), 
            transform=train_transform
        )
        
        # Load full training dataset WITHOUT augmentation for validation
        val_dataset_base = datasets.ImageFolder(
            str(train_dir), 
            transform=test_transform
        )
        
        # Proper split: train -> train/val, keep test separate
        n_train = len(full_train_dataset)
        train_size = int(n_train * self.config.data.train_split)
        val_size = int(n_train * self.config.data.val_split)
        # Remaining goes to test if we're combining with provided test set
        
        # Create indices and split
        indices = np.arange(n_train)
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        
        # Create subsets (use different base datasets!)
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(val_dataset_base, val_indices)
        
        # Create test dataset (use provided test set)
        test_dir = self.dataset_path / "Testing"
        test_dataset = datasets.ImageFolder(str(test_dir), transform=test_transform)
        
        # Class weights for balancing (computed on train subset)
        train_targets = np.array([full_train_dataset.targets[i] for i in train_indices])
        class_counts = np.bincount(train_targets, minlength=self.num_classes)
        class_weights = len(train_subset) / (self.num_classes * (class_counts + 1e-8))
        sample_weights = class_weights[train_targets]
        
        # Create samplers
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        # Log class distribution
        self._log_distribution(train_targets, "Training Set")
        val_targets = np.array([full_train_dataset.targets[i] for i in val_indices])
        self._log_distribution(val_targets, "Validation Set")
        test_targets = np.array(test_dataset.targets)
        self._log_distribution(test_targets, "Test Set")
        
        return train_loader, val_loader, test_loader

    def _get_kaggle_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        from src.kaggle_loader import KaggleDirectDataset
        import torch
        
        train_files, test_files = self.kaggle_loader.get_train_test_split(seed=self.seed)
        
        train_dataset = KaggleDirectDataset(
            self.kaggle_loader, train_files, 
            img_size=(self.config.data.img_size, self.config.data.img_size),
            augment=self.config.data.enable_augmentation, 
            normalize=True
        )
        
        test_dataset_full = KaggleDirectDataset(
            self.kaggle_loader, test_files, 
            img_size=(self.config.data.img_size, self.config.data.img_size),
            augment=False, 
            normalize=True
        )
        
        # Split test into val/test
        val_size = int(len(test_dataset_full) * 0.5)
        test_size = len(test_dataset_full) - val_size
        
        val_dataset, test_dataset = torch.utils.data.random_split(
            test_dataset_full, 
            [val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # Class weighting
        train_targets = np.array([sample[1] for sample in train_dataset.samples])
        class_counts = np.bincount(train_targets, minlength=self.num_classes)
        class_weights = len(train_dataset) / (self.num_classes * (class_counts + 1e-8))
        sample_weights = class_weights[train_targets]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler, 
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        self._log_distribution(train_targets, "Training Set")
        
        return train_loader, val_loader, test_loader
    
    def _log_distribution(self, targets, split_name):
        """Log class distribution for a split."""
        unique, counts = np.unique(targets, return_counts=True)
        dist = {self.class_names[i]: int(counts[idx]) 
                for idx, i in enumerate(unique)}
        print(f"{split_name} distribution: {dist}")


"""Model factory and utilities."""