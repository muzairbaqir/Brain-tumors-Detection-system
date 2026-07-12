from src.config import DataConfig

import os
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import requests

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: kaggle library not installed. Install with: pip install kaggle")


class KaggleDirectLoader:
    """
    Load Kaggle Brain Tumor MRI dataset directly without downloading
    
    Features:
    - Stream data directly from Kaggle
    - No local storage needed
    - Cache frequently used files
    - Support for train/validation/test splits
    """
    
    DATASET_NAME = "masoudnickparvar/brain-tumor-mri-dataset"
    CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
    
    def __init__(self, cache_dir: Optional[str] = None, use_cache: bool = True):
        """
        Initialize Kaggle loader
        
        Args:
            cache_dir: Directory to cache downloaded files (optional)
            use_cache: Whether to use local cache if available
        """
        if not KAGGLE_AVAILABLE:
            raise ImportError(
                "Kaggle API not installed. Run: pip install kaggle\n"
                "Then authenticate: kaggle auth --credentials-file <path>"
            )
        
        self.api = KaggleApi()
        self.api.authenticate()
        
        self.cache_dir = Path(cache_dir or "./kaggle_cache")
        self.use_cache = use_cache
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.file_cache: Dict[str, bytes] = {}  # In-memory cache
        
    def list_dataset_files(self) -> List[str]:
        """List all files in the dataset"""
        files = self.api.dataset_list_files(self.DATASET_NAME).files
        return [f.name for f in files]
    
    def _get_cache_path(self, filename: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / filename
    
    def _load_file_bytes(self, filename: str, force_download: bool = False) -> bytes:
        """
        Load file as bytes from Kaggle (with caching)
        
        Args:
            filename: File path in dataset
            force_download: Force download even if cached
            
        Returns:
            File contents as bytes
        """
        # Check memory cache
        if filename in self.file_cache:
            return self.file_cache[filename]
        
        # Check disk cache
        if self.use_cache:
            cache_path = self._get_cache_path(filename)
            if cache_path.exists() and not force_download:
                data = cache_path.read_bytes()
                self.file_cache[filename] = data  # Store in memory
                return data
        
        # Download from Kaggle
        print(f"⬇️  Downloading from Kaggle: {filename}")
        
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.api.dataset_download_file(
                self.DATASET_NAME,
                filename,
                path=temp_dir,
                quiet=True
            )
            
            downloaded_files = os.listdir(temp_dir)
            if not downloaded_files:
                raise FileNotFoundError(f"Failed to download {filename} from Kaggle")
                
            downloaded_file_path = os.path.join(temp_dir, downloaded_files[0])
            
            if downloaded_file_path.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(downloaded_file_path, 'r') as zip_ref:
                    zip_infos = zip_ref.infolist()
                    data = zip_ref.read(zip_infos[0].filename)
            else:
                with open(downloaded_file_path, 'rb') as f:
                    data = f.read()
        
        # Cache locally if enabled
        if self.use_cache:
            cache_path = self._get_cache_path(filename)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
        
        # Cache in memory
        self.file_cache[filename] = data
        
        return data
    
    def load_image_from_path(self, path: str) -> np.ndarray:
        """
        Load a single image from dataset path
        
        Args:
            path: Path to image in dataset (e.g., "Training/glioma/image_001.jpg")
            
        Returns:
            Image as numpy array (H, W, 3)
        """
        data = self._load_file_bytes(path)
        image = Image.open(io.BytesIO(data)).convert('RGB')
        return np.array(image)
    
    def get_train_test_split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Get train/test split of dataset files
        
        Args:
            train_ratio: Fraction for training (0.8 = 80% train, 20% test)
            seed: Random seed for reproducibility
            
        Returns:
            (train_files, test_files) where each is {class_name: [file_paths]}
        """
        all_files = self.list_dataset_files()
        
        # Filter to only images
        image_files = [f for f in all_files if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Group by class
        files_by_class = {cls: [] for cls in self.CLASS_NAMES}
        
        for img_file in image_files:
            for cls in self.CLASS_NAMES:
                if f"/{cls}/" in img_file or f"_{cls}_" in img_file:
                    files_by_class[cls].append(img_file)
                    break
        
        # Split each class
        np.random.seed(seed)
        train_files = {cls: [] for cls in self.CLASS_NAMES}
        test_files = {cls: [] for cls in self.CLASS_NAMES}
        
        for cls in self.CLASS_NAMES:
            files = files_by_class[cls]
            indices = np.arange(len(files))
            np.random.shuffle(indices)
            
            split_idx = int(len(files) * train_ratio)
            
            train_files[cls] = [files[i] for i in indices[:split_idx]]
            test_files[cls] = [files[i] for i in indices[split_idx:]]
        
        return train_files, test_files
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            "total_files": len(self.list_dataset_files()),
            "classes": self.CLASS_NAMES,
            "kaggle_dataset": self.DATASET_NAME,
            "cache_dir": str(self.cache_dir) if self.use_cache else None,
        }
        
        try:
            train_files, test_files = self.get_train_test_split()
            stats["train_counts"] = {
                cls: len(files) for cls, files in train_files.items()
            }
            stats["test_counts"] = {
                cls: len(files) for cls, files in test_files.items()
            }
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    def download_full_dataset(self, output_dir: str, extract: bool = True):
        """
        Download entire dataset (if you change your mind)
        
        Args:
            output_dir: Directory to save dataset
            extract: Whether to extract zip files
        """
        print(f"📥 Downloading full dataset to {output_dir}...")
        self.api.dataset_download_files(
            self.DATASET_NAME,
            path=output_dir,
            unzip=extract
        )
        print(f"✅ Download complete!")


class KaggleDirectDataset:
    """
    PyTorch Dataset that loads from Kaggle directly
    
    Usage:
        loader = KaggleDirectLoader()
        train_files, test_files = loader.get_train_test_split()
        dataset = KaggleDirectDataset(loader, train_files, augment=True)
        dataloader = DataLoader(dataset, batch_size=32)
    """
    
    def __init__(
        self,
        kaggle_loader: KaggleDirectLoader,
        files_dict: Dict[str, List[str]],
        img_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        normalize: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            kaggle_loader: KaggleDirectLoader instance
            files_dict: Dict mapping class names to file paths
            img_size: Target image size (H, W)
            augment: Whether to apply augmentations
            normalize: Whether to apply ImageNet normalization
        """
        self.loader = kaggle_loader
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize
        
        # Flatten files list with class indices
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(self.loader.CLASS_NAMES)}
        
        for class_name, files in files_dict.items():
            class_idx = self.class_to_idx[class_name]
            for file_path in files:
                self.samples.append((file_path, class_idx, class_name))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get item by index
        
        Returns:
            (image_array, class_index)
        """
        file_path, class_idx, class_name = self.samples[idx]
        
        # Load image from Kaggle
        image = self.loader.load_image_from_path(file_path)
        
        # Resize
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize(self.img_size, Image.Resampling.BILINEAR)
        image = np.array(image_pil, dtype=np.float32)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Apply augmentations (if training)
        if self.augment:
            image = self._augment(image)
        
        # ImageNet normalization
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image = (image - mean) / std
        
        # Convert to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        return image.astype(np.float32), class_idx
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentations"""
        # Random brightness
        if np.random.rand() > 0.5:
            image = image * np.random.uniform(0.7, 1.3)
            image = np.clip(image, 0, 1)
        
        # Random contrast
        if np.random.rand() > 0.5:
            mean = image.mean()
            image = (image - mean) * np.random.uniform(0.7, 1.3) + mean
            image = np.clip(image, 0, 1)
        
        return image


def create_kaggle_config() -> Dict:
    """
    Create configuration dict for Kaggle direct loading
    
    Returns dict to merge with ExperimentConfig.data
    """
    return {
        "dataset_path": "kaggle://masoudnickparvar/brain-tumor-mri-dataset",
        "use_kaggle_api": True,
        "cache_dir": "./kaggle_cache",
        "use_cache": True,
        "splits": {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1
        }
    }


# Example usage
if __name__ == "__main__":
    print("🔗 Kaggle Direct Loader - Example")
    print("=" * 60)
    
    try:
        # Initialize loader
        loader = KaggleDirectLoader(cache_dir="./kaggle_cache", use_cache=True)
        
        # Get stats
        stats = loader.get_dataset_stats()
        print(f"\n📊 Dataset Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Get train/test split
        train_files, test_files = loader.get_train_test_split()
        
        print(f"\n✅ Successfully connected to Kaggle!")
        print(f"📁 Training images: {sum(len(f) for f in train_files.values())}")
        print(f"📁 Test images: {sum(len(f) for f in test_files.values())}")
        
        # Example: load one image
        if train_files["glioma"]:
            img_path = train_files["glioma"][0]
            print(f"\n🖼️  Loading sample image: {img_path}")
            image = loader.load_image_from_path(img_path)
            print(f"   Shape: {image.shape}, dtype: {image.dtype}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTo set up Kaggle API:")
        print("1. pip install kaggle")
        print("2. Download API key from https://www.kaggle.com/settings/account")
        print("3. Place in ~/.kaggle/kaggle.json")
        print("4. chmod 600 ~/.kaggle/kaggle.json")


"""Data loading, preprocessing, and splitting."""