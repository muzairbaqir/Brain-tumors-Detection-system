from src.config import ExperimentConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training with validation."""
    
    def __init__(self, config, model, optimizer, scheduler, loss_fn, device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        
        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.loss_fn(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = running_loss / total
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            outputs = self.model(imgs)
            loss = self.loss_fn(outputs, labels)
            
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = running_loss / total
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        epochs = self.config.training.epochs
        patience = self.config.training.patience
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(self.optimizer.param_groups[0]['lr'])
            
            # Log
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Scheduler step
            self.scheduler.step()
            
            # Checkpointing
            if self.config.training.save_best_only:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
            else:
                self._save_checkpoint(epoch, is_best=(val_acc > self.best_val_acc))
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
            
            if self.config.training.save_last:
                self._save_checkpoint(epoch, is_best=False, name="last")
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience exceeded)")
                break
        
        logger.info(f"Training complete. Best val acc: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        self._save_history()
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, name: str = "best"):
        """Save model checkpoint."""
        if name == "best":
            ckpt_path = self.checkpoint_dir / "best_model.pt"
        else:
            ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
        }
        
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")
    
    def _save_history(self):
        """Save training history as JSON."""
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_best_checkpoint(self):
        """Load best saved checkpoint."""
        ckpt_path = self.checkpoint_dir / "best_model.pt"
        if not ckpt_path.exists():
            logger.warning(f"No checkpoint found at {ckpt_path}")
            return
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        logger.info(f"Loaded checkpoint from {ckpt_path}")

