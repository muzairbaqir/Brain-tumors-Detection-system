# Brain Tumor Detection: Robustness Thesis Pipeline

A production-level ML pipeline for evaluating the robustness of Swin Transformer on brain tumor MRI classification.

## Research Question

**How robust is Swin Transformer to realistic MRI corruptions and domain shifts? Can we improve robustness through training strategies?**

This thesis investigates:
1. **Baseline robustness** - How does the model perform under realistic MRI corruptions (noise, motion blur, intensity shifts)?
2. **Corruption severity** - Which corruption types hurt performance most?
3. **Robustness improvement** (Phase 2) - Can we train with augmentation/adversarial techniques to improve robustness?

## Project Structure

```
thesis_pipeline/
├── src/
│   ├── config.py           # Configuration management (YAML-based)
│   ├── data.py             # Data loading with proper train/val/test split
│   ├── models.py           # Model factory and utilities
│   ├── training.py         # Training loop with early stopping & checkpointing
│   ├── evaluation.py       # Evaluation metrics and visualization
│   └── robustness.py       # Corruption generators & robustness testing
├── scripts/
│   ├── train.py            # Main training script
│   ├── eval_robustness.py  # Robustness evaluation script
│   └── train_robust.py     # Train with robustness (Phase 2)
├── configs/
│   ├── default.yaml        # Default configuration
│   └── ablation/           # Ablation study configs
├── outputs/                # Experiment results & metrics
├── checkpoints/            # Trained model weights
└── logs/                   # Training logs

```

## Setup

### Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)

### Installation

```bash
# Clone or download the project
cd thesis_pipeline

# Install dependencies
pip install torch torchvision timm scikit-learn matplotlib seaborn numpy pyyaml pillow opencv-python scipy

# Download dataset from Kaggle
# https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
# Extract to a location (e.g., ./data/brain-tumor-mri-dataset)
```

## Usage

### Phase 1: Train Baseline Model

```bash
# Edit config with your dataset path
nano configs/default.yaml
# Update: data.dataset_path = "/path/to/brain-tumor-mri-dataset"

# Run training
python scripts/train.py --config configs/default.yaml
```

**Output:**
- `checkpoints/best_model.pt` - Best model weights
- `checkpoints/history.json` - Training history
- `outputs/evaluation/` - Confusion matrix, classification report, etc.
- `logs/` - Training logs

### Phase 2: Evaluate Robustness (Main Thesis Angle)

```bash
# After training completes
python scripts/eval_robustness.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best_model.pt
```

**Output:**
- `outputs/robustness/robustness_results.json` - Performance under each corruption
- Shows accuracy drop per corruption type and severity level

### Phase 3: Train with Robustness (Optional, for improved model)

```bash
# Train with augmented data to improve robustness
python scripts/train_robust.py --config configs/robustness_training.yaml
```

## Experiment Workflow

### Understanding the Results

After running robustness evaluation, you'll get results like:

```
Clean Accuracy: 0.9200

Gaussian Noise:
  Level 1: Acc=0.9100, Drop=0.0100
  Level 2: Acc=0.8950, Drop=0.0250
  Level 3: Acc=0.8650, Drop=0.0550
  ...

Motion Blur:
  Level 1: Acc=0.8800, Drop=0.0400
  Level 2: Acc=0.8100, Drop=0.1100
  ...
```

**Key Metrics:**
- **Drop**: How much accuracy decreases under corruption
- **Severity Level**: 1-5 (5 is most severe)

### Thesis Analysis Points

Use these results to discuss:

1. **Which corruptions hurt most?** (Motion blur > Gaussian noise?)
2. **Is performance degradation consistent across classes?** (Analyze per-class accuracy drops)
3. **Are attention maps robust?** (Extract Swin attention during robustness eval)
4. **Compare to baselines:** Train ResNet50, DenseNet121 with same config
5. **Can we improve robustness?** Train with synthetic corruptions as augmentation

## Configuration System

Configs are YAML-based and fully parameterized. No hardcoded values.

### Creating Custom Configs

```bash
# Copy and modify
cp configs/default.yaml configs/my_experiment.yaml
nano configs/my_experiment.yaml

# Then run with it
python scripts/train.py --config configs/my_experiment.yaml
```

### Ablation Studies

```yaml
# configs/ablation/swin_small.yaml
model:
  name: "swin_small_patch4_window7_224"
  
# configs/ablation/higher_lr.yaml
training:
  learning_rate: 5.0e-5
```

Run multiple experiments and compare results.

## Key Methodological Improvements over Notebook

✅ **Proper train/val/test split** - No test set leakage
✅ **Config-driven** - Reproducible, no hardcoding
✅ **Early stopping** - Stops when validation accuracy plateaus
✅ **Checkpointing** - Saves best model automatically
✅ **Class balancing** - Handles imbalanced tumor types
✅ **Reproducibility** - Fixed seed, deterministic training
✅ **Logging** - Full experiment tracking
✅ **Robustness focus** - Corruption generators for thesis-level work

## Reproduction Checklist

To reproduce your thesis results:

```bash
# 1. Set random seed (done in config)
# 2. Use same dataset split (use default train/val/test from config)
# 3. Load config from saved YAML
# 4. Report all metrics (saved in outputs/)
# 5. Include model checkpoint for reproducibility
```

All configs are automatically saved to `logs/<experiment_name>_config.yaml` for reference.

## Expected Timeline

- **Weeks 1-4:** Train baseline model, establish clean accuracy baseline
- **Weeks 5-8:** Robustness evaluation - test all corruption types
- **Weeks 9-12:** Analysis & comparison - which corruptions matter most?
- **Weeks 13-16:** Robustness improvement - train with augmentation, measure improvement
- **Weeks 17-20:** Writing & visualization - create plots for thesis

## Troubleshooting

### "dataset_path does not exist"
- Download Kaggle dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Update `configs/default.yaml` with actual path

### GPU out of memory
- Reduce `batch_size` in config (e.g., 16 instead of 32)
- Use smaller model: `swin_tiny_patch4_window7_224` (already default)

### Training too slow
- Use GPU: `torch.cuda.is_available()` should return True
- Reduce `num_workers` in config if disk I/O is slow

### Results not reproducing
- Check that `seed` is set in config
- Verify `torch.backends.cudnn.deterministic = True` is set
- Use same dataset splits (don't change train_split/val_split)

## Next Steps for Your Thesis

1. **Get baseline:** Run Phase 1 & 2, document clean accuracy
2. **Analyze robustness:** Create plots showing accuracy vs. corruption severity
3. **Investigate failure modes:** Which classes/tumors are most affected?
4. **Compare architectures:** Run same pipeline with ResNet, DenseNet
5. **Improve robustness:** Train Phase 3 with augmentation, measure improvement
6. **Write thesis:** Use results & plots from outputs/ directory

## References

- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [ImageNet-C: Robustness Corruption Benchmark](https://arxiv.org/abs/1903.12261) (inspiration for corruption types)
- [Medical Image Analysis with Deep Learning](https://arxiv.org/abs/2102.14662)

---

**Last Updated:** 2026
**Maintainer:** Your Name
