"""Training orchestration and epoch management."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy
import time
from pathlib import Path

from data.genetic_distance import labels_to_soft_labels
from utils.metrics import MetricsTracker


class Trainer:
    """Handles model training and validation."""
    
    def __init__(self, model, dataloaders, config, device, soft_label_matrix=None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            dataloaders: Dictionary with 'train' and 'val' dataloaders
            config: Configuration dictionary
            device: Device to train on
            soft_label_matrix: Optional soft label matrix for dist_loss
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.config = config
        self.device = device
        self.soft_label_matrix = soft_label_matrix
        
        # Training parameters
        self.num_epochs = config['training']['num_epochs']
        self.loss_type = config['training']['loss_type']
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.best_model_wts = None
        
    def _setup_optimizer(self):
        """Setup optimizer from config."""
        opt_config = self.config['training']['optimizer']
        
        if opt_config['type'] == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config['momentum']
            )
        elif opt_config['type'] == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler from config."""
        sched_config = self.config['training']['scheduler']
        
        if sched_config['type'] == 'StepLR':
            return StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_config['type']}")
    
    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        dataloader = self.dataloaders['train']
        num_batches = len(dataloader)
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)
            
            # Compute loss
            if self.loss_type == 'score_loss':
                loss = self.criterion(outputs, labels)
            elif self.loss_type == 'dist_loss':
                # Convert labels to soft labels
                soft_labels = labels_to_soft_labels(labels.cpu(), self.soft_label_matrix)
                soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                loss = self.criterion(outputs, soft_labels)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            
            # Progress indicator
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                progress = (batch_idx + 1) / num_batches * 100
                print(f'\rProgress: {progress:.0f}%', end='')
        
        print()  # New line after progress
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def _validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        dataloader = self.dataloaders['val']
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                batch_size = images.size(0)
                total_samples += batch_size
                
                # Forward pass
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                # Compute loss
                if self.loss_type == 'score_loss':
                    loss = self.criterion(outputs, labels)
                elif self.loss_type == 'dist_loss':
                    soft_labels = labels_to_soft_labels(labels.cpu(), self.soft_label_matrix)
                    soft_labels = torch.from_numpy(soft_labels).float().to(self.device)
                    loss = self.criterion(outputs, soft_labels)
                
                # Track metrics
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return epoch_loss, epoch_acc.item()
    
    def train(self):
        """
        Run full training loop.
        
        Returns:
            Trained model with best weights loaded
        """
        print(f"\n{'='*60}")
        print(f"TRAINING BEGIN: {self.config['experiment']['name']}")
        print(f"Architecture: {self.config['model']['architecture']}")
        print(f"Loss type: {self.loss_type}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        total_start = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)
            
            # Train
            print("Training...")
            train_loss, train_acc = self._train_epoch()
            print(f"  Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            
            # Validate
            print("Validating...")
            val_loss, val_acc = self._validate_epoch()
            print(f"  Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Track metrics
            self.metrics_tracker.update(epoch + 1, train_loss, train_acc, 
                                       val_loss, val_acc)
            
            # Save best model
            if val_acc > self.metrics_tracker.best_val_acc:
                print(f"  * New best validation accuracy: {val_acc:.4f}")
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
            
            epoch_time = time.time() - epoch_start
            print(f"  Time: {epoch_time / 60:.2f} min\n")
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"Total time: {total_time / 60:.2f} min")
        print(f"Best validation accuracy: {self.metrics_tracker.best_val_acc:.4f} "
              f"(Epoch {self.metrics_tracker.best_epoch})")
        print(f"{'='*60}\n")
        
        # Load best weights
        if self.best_model_wts is not None:
            self.model.load_state_dict(self.best_model_wts)
        
        return self.model
    
    def get_metrics(self):
        """Get all tracked metrics."""
        return self.metrics_tracker.get_all()
    
    def get_best_metrics(self):
        """Get best validation metrics."""
        return self.metrics_tracker.get_best()
