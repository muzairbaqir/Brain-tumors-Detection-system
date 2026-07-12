import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def _get_classes(dataset_path: Path):
    train_dir = dataset_path / "Training"
    if not train_dir.exists():
        return []
    return sorted([d for d in os.listdir(train_dir) if os.path.isdir(train_dir / d)])

def plot_class_distribution(dataset_path: str, output_dir: str):
    """Plot bar chart of class distributions for train/test."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classes = _get_classes(dataset_path)
    if not classes:
        print("Warning: Could not find training classes for visualization.")
        return

    train_counts = []
    test_counts = []
    
    for cls in classes:
        train_len = len(os.listdir(dataset_path / "Training" / cls))
        test_len = len(os.listdir(dataset_path / "Testing" / cls))
        train_counts.append(train_len)
        test_counts.append(test_len)
        
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_counts, width, label='Training')
    rects2 = ax.bar(x + width/2, test_counts, width, label='Testing')
    
    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Class Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()

def plot_raw_samples(dataset_path: str, output_dir: str, num_samples_per_class: int = 4):
    """Plot a grid of raw, unaugmented images directly from the dataset folder."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classes = _get_classes(dataset_path)
    if not classes:
        return
        
    fig, axes = plt.subplots(len(classes), num_samples_per_class, figsize=(num_samples_per_class*3, len(classes)*3))
    
    for i, cls in enumerate(classes):
        cls_dir = dataset_path / "Training" / cls
        images = os.listdir(cls_dir)
        sampled_images = random.sample(images, min(len(images), num_samples_per_class))
        
        for j, img_name in enumerate(sampled_images):
            img_path = cls_dir / img_name
            img = Image.open(img_path)
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
            ax.axis('off')
            if j == 0:
                ax.set_title(cls, fontsize=14, loc='left')
                
    plt.tight_layout()
    plt.savefig(output_dir / "raw_samples_grid.png", dpi=200)
    plt.close()

def plot_augmentation_effects(dataset, output_dir: str):
    """Plot before/after augmentation for a few random samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We need the dataset path to read original image
    if hasattr(dataset, 'dataset_path'):
        dataset_path = Path(dataset.dataset_path)
    elif hasattr(dataset, 'config') and hasattr(dataset.config.data, 'dataset_path'):
        dataset_path = Path(dataset.config.data.dataset_path)
    else:
        return
        
    classes = _get_classes(dataset_path)
    if not classes:
        return

    # Create a simplified transform without Normalization for visualization
    aug_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(dataset.config.data.aug_rotation if hasattr(dataset.config.data, 'aug_rotation') else 10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])
    
    fig, axes = plt.subplots(4, 2, figsize=(8, 16))
    
    for i in range(4):
        # Pick random class and image
        cls = random.choice(classes)
        cls_dir = dataset_path / "Training" / cls
        img_name = random.choice(os.listdir(cls_dir))
        
        # Load raw
        raw_img = Image.open(cls_dir / img_name).convert('RGB')
        
        # Apply aug
        aug_img = aug_transform(raw_img)
        
        # Plot raw
        axes[i, 0].imshow(raw_img)
        axes[i, 0].set_title(f"Original ({cls})")
        axes[i, 0].axis('off')
        
        # Plot aug
        axes[i, 1].imshow(aug_img)
        axes[i, 1].set_title("Augmented")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / "augmentation_comparison.png", dpi=150)
    plt.close()

def plot_tensor_distributions(dataloader, output_dir: str):
    """Plot histogram of tensor values after normalization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get one batch
    images, _ = next(iter(dataloader))
    
    # Flatten and plot
    vals = images.numpy().flatten()
    
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=100, color='purple', alpha=0.7)
    plt.title('Pixel Value Distribution (After Normalization)')
    plt.xlabel('Tensor Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "tensor_distributions.png", dpi=150)
    plt.close()

def plot_misclassified(images, true_labels, pred_labels, class_names, output_dir: str, num_samples: int = 16):
    """Plot a grid of misclassified images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find incorrect indices
    incorrect_idx = np.where(true_labels != pred_labels)[0]
    
    if len(incorrect_idx) == 0:
        # Create a dummy image indicating perfect score
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Perfect Classification!\nNo misclassified samples.", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.savefig(output_dir / "misclassified_samples.png", dpi=150)
        plt.close()
        return
        
    # Sample up to num_samples
    if len(incorrect_idx) > num_samples:
        incorrect_idx = np.random.choice(incorrect_idx, num_samples, replace=False)
        
    n_cols = 4
    n_rows = int(np.ceil(len(incorrect_idx) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3.5))
    axes = axes.flatten() if n_rows > 1 else (axes if isinstance(axes, np.ndarray) else [axes])
    
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    for i, idx in enumerate(incorrect_idx):
        if i >= len(axes):
            break
            
        # Get image and un-normalize
        img = images[idx].copy()
        img = img.transpose(1, 2, 0) # CHW to HWC
        img = img * imagenet_std + imagenet_mean
        img = np.clip(img, 0, 1)
        
        true_cls = class_names[true_labels[idx]]
        pred_cls = class_names[pred_labels[idx]]
        
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"True: {true_cls}\nPred: {pred_cls}", 
                     color='red' if true_cls != pred_cls else 'black',
                     fontsize=10)
        ax.axis('off')
        
    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / "misclassified_samples.png", dpi=150)
    plt.close()

def plot_latent_space(features, labels, class_names, output_dir: str):
    """Plot PCA and t-SNE 2D projections of the high-dimensional feature space."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(features)-1), max_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, class_name in enumerate(class_names):
        idx = np.where(labels == i)[0]
        axes[0].scatter(pca_result[idx, 0], pca_result[idx, 1], c=colors[i % len(colors)], label=class_name, alpha=0.7, s=20)
        axes[1].scatter(tsne_result[idx, 0], tsne_result[idx, 1], c=colors[i % len(colors)], label=class_name, alpha=0.7, s=20)
        
    axes[0].set_title('PCA Latent Space')
    axes[0].legend()
    axes[1].set_title('t-SNE Latent Space')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "latent_space_clusters.png", dpi=200)
    plt.close()

def plot_saliency(model, imgs, labels, class_names, output_dir: str, device):
    """Generate and plot gradient-based Saliency Maps to see where the model looks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Take up to 4 images
    num_samples = min(4, len(imgs))
    imgs = imgs[:num_samples].clone().detach().to(device)
    labels = labels[:num_samples].cpu().numpy()
    
    # Enable gradients for input images
    imgs.requires_grad = True
    
    # Forward pass
    outputs = model(imgs)
    
    # Backward pass for each image to get saliency
    saliencies = []
    for i in range(num_samples):
        model.zero_grad()
        score = outputs[i, labels[i]]
        score.backward(retain_graph=True)
        
        # Saliency is the absolute value of the gradient
        saliency = imgs.grad[i].data.abs().cpu().numpy()
        # Pool across color channels to get a 2D heatmap
        saliency = np.max(saliency, axis=0)
        saliencies.append(saliency)
        
        # Reset grad for next image
        imgs.grad.data.zero_()
        
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(num_samples):
        # Original Image
        img_vis = imgs[i].detach().cpu().numpy().transpose(1, 2, 0)
        img_vis = img_vis * imagenet_std + imagenet_mean
        img_vis = np.clip(img_vis, 0, 1)
        
        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title(f"Original ({class_names[labels[i]]})")
        axes[i, 0].axis('off')
        
        # Saliency Map overlay
        axes[i, 1].imshow(img_vis)
        axes[i, 1].imshow(saliencies[i], cmap='hot', alpha=0.5)
        axes[i, 1].set_title("Saliency Map")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / "saliency_maps.png", dpi=150)
    plt.close()


"""Evaluation metrics and visualization."""