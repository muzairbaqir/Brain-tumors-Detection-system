import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter

class MedicalCorruptions:
    @staticmethod
    def apply_corruption(image: torch.Tensor, corruption_name: str, severity: int) -> torch.Tensor:
        """
        Apply a specific corruption at a given severity (1-5) to an image tensor.
        image: PyTorch tensor of shape (C, H, W) in [0, 1]
        """
        # Convert tensor to numpy for easier manipulation, shape (H, W, C)
        img_np = image.permute(1, 2, 0).cpu().numpy()
        
        if corruption_name == "gaussian_noise":
            img_np = MedicalCorruptions._gaussian_noise(img_np, severity)
        elif corruption_name == "motion_blur":
            img_np = MedicalCorruptions._motion_blur(img_np, severity)
        elif corruption_name == "elastic_deform":
            img_np = MedicalCorruptions._elastic_deform(img_np, severity)
        elif corruption_name == "intensity_shift":
            img_np = MedicalCorruptions._intensity_shift(img_np, severity)
        elif corruption_name == "spike_noise":
            img_np = MedicalCorruptions._spike_noise(img_np, severity)
        else:
            raise ValueError(f"Unknown corruption: {corruption_name}")
            
        # Ensure values are valid
        img_np = np.clip(img_np, 0.0, 1.0)
        
        # Back to tensor
        return torch.from_numpy(img_np).permute(2, 0, 1).float()

    @staticmethod
    def _gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
        # standard deviations for severity 1-5
        stds = [0.04, 0.08, 0.12, 0.18, 0.26]
        noise = np.random.normal(scale=stds[severity - 1], size=img.shape)
        return img + noise

    @staticmethod
    def _motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
        # kernel sizes for severity 1-5
        kernels = [3, 5, 7, 9, 11]
        k = kernels[severity - 1]
        
        # Create a motion blur kernel (horizontal)
        kernel_motion_blur = np.zeros((k, k))
        kernel_motion_blur[int((k-1)/2), :] = np.ones(k)
        kernel_motion_blur = kernel_motion_blur / k
        
        from scipy.signal import convolve2d
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = convolve2d(img[:, :, c], kernel_motion_blur, mode='same', boundary='symm')
        return out

    @staticmethod
    def _elastic_deform(img: np.ndarray, severity: int) -> np.ndarray:
        # Elastic deformation parameters
        alphas = [10, 20, 30, 40, 50]
        sigmas = [4, 4, 4, 5, 5]
        
        alpha = alphas[severity - 1]
        sigma = sigmas[severity - 1]
        random_state = np.random.RandomState(None)
        
        shape = img.shape[:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = map_coordinates(img[:, :, c], indices, order=1, mode='reflect').reshape(shape)
            
        return out

    @staticmethod
    def _intensity_shift(img: np.ndarray, severity: int) -> np.ndarray:
        # shifts for severity 1-5
        shifts = [0.05, 0.1, 0.2, 0.3, 0.4]
        shift = shifts[severity - 1]
        
        # Apply shift (simulate different contrast agent)
        out = img + shift
        return out

    @staticmethod
    def _spike_noise(img: np.ndarray, severity: int) -> np.ndarray:
        # num spikes for severity 1-5
        probs = [0.01, 0.02, 0.03, 0.05, 0.08]
        prob = probs[severity - 1]
        
        out = np.copy(img)
        mask = np.random.rand(*img.shape[:2]) < prob
        for c in range(img.shape[2]):
            out[:, :, c][mask] = 1.0 # White spikes
            
        return out
