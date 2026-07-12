# Quick Start Guide

## 30-Second Setup

1. **Edit the config with your dataset path:**
   ```bash
   nano configs/default.yaml
   # Change: dataset_path: "/actual/path/to/brain-tumor-mri-dataset"
   ```

2. **Run training:**
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```

3. **Evaluate robustness (MAIN THESIS WORK):**
   ```bash
   python scripts/eval_robustness.py \
     --config configs/default.yaml \
     --checkpoint checkpoints/best_model.pt
   ```

That's it. Results appear in `outputs/`.

---

## What Gets Generated

```
outputs/
├── robustness/
│   └── robustness_results.json          # Main results: accuracy per corruption
├── evaluation/
│   ├── confusion_matrix.png
│   ├── classification_report.png
│   └── ...
checkpoints/
├── best_model.pt                        # Your trained model
├── history.json                         # Training curves (plot this for thesis)
└── training_history.png
logs/
├── swin_tumor_robustness_study.log     # Full training log
└── swin_tumor_robustness_study_config.yaml  # Saved config for reproducibility
```

---

## Your Thesis Workflow

### Week 1-2: Baseline Training
```bash
# Train once, get baseline accuracy
python scripts/train.py --config configs/default.yaml

# Check results:
cat outputs/evaluation/classification_report.png
# Look for: Overall accuracy ~90%+ on clean data
```

### Week 3-4: Robustness Analysis (CORE CONTRIBUTION)
```bash
# Run robustness evaluation
python scripts/eval_robustness.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best_model.pt

# Check results:
cat outputs/robustness/robustness_results.json
# Analyze: Which corruptions hurt most?
# Create thesis plots showing accuracy drops
```

### Week 5-6: Comparison Study (Optional but Strong)
```bash
# Create config for ResNet baseline
cp configs/default.yaml configs/resnet50_baseline.yaml
# Edit: model.name: "resnet50"

# Train ResNet
python scripts/train.py --config configs/resnet50_baseline.yaml

# Compare robustness
python scripts/eval_robustness.py \
  --config configs/resnet50_baseline.yaml \
  --checkpoint checkpoints/best_model.pt

# Compare plots: Swin vs ResNet robustness
```

### Week 7-8: Robustness Improvement (Phase 2 - Optional)
```bash
# Train with corruption augmentation
python scripts/train_robust.py \
  --config configs/default.yaml \
  --pretrained checkpoints/best_model.pt

# Compare new model robustness
python scripts/eval_robustness.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/robust/best_model.pt
```

---

## Example Commands for Different Experiments

### Test with Higher Learning Rate
```bash
# Create config
cp configs/default.yaml configs/ablation/high_lr.yaml

# Edit configs/ablation/high_lr.yaml
# Change: training.learning_rate: 5.0e-5

# Run
python scripts/train.py --config configs/ablation/high_lr.yaml

# Compare to baseline
```

### Test with Different Model
```bash
# Create config
cp configs/default.yaml configs/ablation/swin_base.yaml

# Edit configs/ablation/swin_base.yaml
# Change: model.name: "swin_base_patch4_window7_224"

# Run
python scripts/train.py --config configs/ablation/swin_base.yaml
```

### Test with Different Augmentation
```bash
# Create config
cp configs/default.yaml configs/ablation/aggressive_aug.yaml

# Edit:
# data.aug_rotation: 20  (more aggressive)
# data.aug_brightness: 0.2

# Run
python scripts/train.py --config configs/ablation/aggressive_aug.yaml
```

---

## Extracting Results for Thesis

### Get Clean Baseline Accuracy
```bash
python -c "
import json
with open('outputs/evaluation/classification_report.png') as f:
    # (Or directly from logs)
    print('Check: outputs/robustness/robustness_results.json')
"

# Better: Look at the JSON directly
cat outputs/robustness/robustness_results.json | grep accuracy
```

### Create Comparison Table
```bash
# Run all experiments, then:
python -c "
import json
import pandas as pd

results = {}
for exp in ['swin', 'resnet', 'densenet']:
    with open(f'outputs/{exp}/robustness_results.json') as f:
        results[exp] = json.load(f)

# Print comparison
for exp, data in results.items():
    print(f'{exp}: Clean Acc = {data[\"clean\"][\"accuracy\"]:.4f}')
"
```

### Plot Robustness Degradation
```bash
# Use the saved training_history.png
# + custom script to plot corruption severity curves
python -c "
import json
import matplotlib.pyplot as plt

with open('outputs/robustness/robustness_results.json') as f:
    results = json.load(f)

# Plot accuracy vs corruption severity
corruptions = results['corruptions']
for corruption_type, severity_data in corruptions.items():
    accs = [severity_data[f'level_{i}']['accuracy'] for i in range(1,6)]
    plt.plot(range(1,6), accs, marker='o', label=corruption_type)

plt.xlabel('Corruption Severity')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('robustness_curves.png')
"
```

---

## Debugging

### Model not improving
- Check: `logs/swin_tumor_robustness_study.log`
- Look for: "Train Loss" should decrease over epochs
- If not: learning rate too high/low, try 1e-5 or 5e-5

### Out of memory
- Reduce batch_size: `data.batch_size: 16`
- Or use smaller model: `model.name: "swin_tiny_patch4_window7_224"`

### Dataset path issues
```bash
# Verify structure:
ls -la /path/to/brain-tumor-mri-dataset/
# Should show: Training/ Testing/

ls -la /path/to/brain-tumor-mri-dataset/Training/
# Should show: glioma/ meningioma/ no_tumor/ pituitary/
```

### Want to use GPU?
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## Your Thesis Title Ideas

1. "Robustness of Vision Transformers to MRI Acquisition Artifacts: A Swin Transformer Study"
2. "Evaluating Domain Robustness of Swin Transformers for Medical Image Classification"
3. "Realistic Corruption Robustness in Brain Tumor MRI Classification with Vision Transformers"
4. "Improving Robustness of Swin Transformers Through Corruption-Aware Training"

---

## Next Step

Edit `configs/default.yaml` with your dataset path, then run:
```bash
python scripts/train.py --config configs/default.yaml
```

Come back with results!
