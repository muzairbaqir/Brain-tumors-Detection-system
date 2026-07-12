
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, roc_auc_score, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config, output_dir: Path = None):
        self.config = config
        self.output_dir = output_dir or (Path(config.output_dir) / "model_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: list,
        device: torch.device,
    ) -> dict:
        """
        Full evaluation pipeline.
        
        Returns:
            Dictionary with metrics and predictions.
        """
        model.eval()
        
        all_imgs = []
        all_preds = []
        all_labels = []
        all_probs = []
        all_features = []
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                
                # Extract features if supported
                if hasattr(model, 'forward_features') and hasattr(model, 'forward_head'):
                    raw_features = model.forward_features(imgs)
                    features = model.forward_head(raw_features, pre_logits=True)
                    outputs = model.forward_head(raw_features)
                    all_features.append(features.cpu().numpy())
                else:
                    outputs = model(imgs)
                    
                probs = torch.softmax(outputs, dim=1)
                
                all_imgs.append(imgs.cpu())
                all_probs.append(probs.cpu())
                all_preds.append(outputs.argmax(1).cpu())
                all_labels.append(labels)
            
            all_imgs = torch.cat(all_imgs).numpy()
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            all_probs = torch.cat(all_probs).numpy()
            if all_features:
                all_features = np.concatenate(all_features, axis=0)
        
        # Compute metrics
        results = {
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
            "probabilities": all_probs.tolist(),
        }
        
        # Accuracy
        accuracy = (all_preds == all_labels).mean()
        results["accuracy"] = float(accuracy)
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            output_dict=True,
        )
        results["classification_report"] = report
        logger.info(f"\nClassification Report:\n{classification_report(all_labels, all_preds, target_names=class_names)}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        results["confusion_matrix"] = cm.tolist()
        
        # ROC-AUC (multiclass)
        try:
            y_bin = label_binarize(all_labels, classes=range(len(class_names)))
            roc_auc = {}
            for i, class_name in enumerate(class_names):
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
                    auc_score = auc(fpr, tpr)
                    roc_auc[class_name] = float(auc_score)
                except Exception as e:
                    logger.warning(f"Could not compute ROC-AUC for class {class_name}: {e}")
            
            results["roc_auc"] = roc_auc
            logger.info(f"ROC-AUC scores: {roc_auc}")
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
        
        # Visualizations
        self._plot_confusion_matrix(cm, class_names)
        self._plot_classification_report(report, class_names)
        
        try:
            from src.visualize import plot_misclassified, plot_latent_space, plot_saliency
            plot_misclassified(all_imgs, all_labels, all_preds, class_names, self.output_dir)
            logger.info(f"Saved misclassified samples to {self.output_dir}")
            
            if len(all_features) > 0:
                logger.info("Generating PCA and t-SNE latent space clusters...")
                plot_latent_space(all_features, all_labels, class_names, self.output_dir)
                
            # 7. Plot Calibration curves (Reliability Diagram)
            self._plot_calibration_curve(all_labels, all_probs, class_names)
            
            # 8. Saliency Maps
            logger.info("Generating Saliency Maps...")
            # We need original raw tensor imgs to calculate gradients
            sample_imgs, sample_labels = next(iter(test_loader))
            sample_imgs = sample_imgs.to(device)
            plot_saliency(model, sample_imgs, sample_labels, class_names, self.output_dir, device)
            
        except Exception as e:
            logger.warning(f"Could not plot advanced visualizations: {e}")
        
        return results

    def _plot_calibration_curve(self, y_true: np.ndarray, y_probs: np.ndarray, class_names: list):
        """Plot calibration curve (Reliability Diagram) and calculate Brier Score."""
        y_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        brier_scores = {}
        for i, class_name in enumerate(class_names):
            prob_pos = y_probs[:, i]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_bin[:, i], prob_pos, n_bins=10)
            
            bscore = brier_score_loss(y_bin[:, i], prob_pos)
            brier_scores[class_name] = bscore
            
            ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label=f"{class_name} (Brier={bscore:.3f})")
                    
        ax.set_ylabel("Fraction of positives")
        ax.set_xlabel("Mean predicted value")
        ax.set_title("Calibration Curve (Reliability Diagram)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        save_path = self.output_dir / "calibration_curve.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved calibration curve to {save_path}")
        logger.info(f"Brier Scores: {brier_scores}")
        plt.close()
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    def _plot_classification_report(self, report: dict, class_names: list):
        """Plot per-class metrics."""
        metrics = ['precision', 'recall', 'f1-score']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for ax, metric in zip(axes, metrics):
            values = [report[cn][metric] for cn in class_names]
            ax.bar(class_names, values, color='steelblue')
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim([0, 1.1])
            ax.set_title(f'{metric.capitalize()} per Class')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.output_dir / "classification_report.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved classification report to {save_path}")
        plt.close()


def plot_training_history(history: dict, output_dir: Path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history["train_loss"], label='Train', marker='o')
    axes[0].plot(history["val_loss"], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history["train_acc"], label='Train', marker='o')
    axes[1].plot(history["val_acc"], label='Val', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved training history to {save_path}")
    plt.close()
