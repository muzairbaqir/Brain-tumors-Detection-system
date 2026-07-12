import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from src.models import build_model
from src.data import get_dataloaders
from src.robustness import MedicalCorruptions
from tqdm import tqdm

def run_robustness_test(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config['model']['name'], num_classes=4).to(device)
    checkpoint = torch.load('brain_tumor_focal_loss/outputs/model_results/checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    _, _, test_loader = get_dataloaders(config)
    corruptions = ['gaussian_noise', 'motion_blur', 'intensity_shift', 'spike_noise']
    severities = [1, 2, 3, 4, 5]
    results = {c: [] for c in corruptions}

    for corr in corruptions:
        print(f'Testing corruption: {corr}')
        for sev in severities:
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in tqdm(test_loader, leave=False):
                    imgs_np = imgs.numpy()
                    corr_func = getattr(MedicalCorruptions, corr)
                    corrupted_imgs = np.array([corr_func(img.squeeze(), sev) for img in imgs_np])
                    corrupted_imgs = torch.from_numpy(corrupted_imgs).unsqueeze(1).to(device)
                    
                    outputs = model(corrupted_imgs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
            
            acc = correct / total
            results[corr].append(acc)

    plt.figure(figsize=(10, 6))
    for corr in corruptions:
        plt.plot(severities, results[corr], marker='o', label=corr)
    plt.title('Model Robustness to MRI Corruptions')
    plt.xlabel('Severity Level')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('brain_tumor_focal_loss/outputs/model_results/robustness_degradation.png')
    print('Robustness plot saved to outputs/model_results/robustness_degradation.png')

if __name__ == '__main__':
    run_robustness_test('master_config.yaml')
