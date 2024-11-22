from typing import Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: Optional[np.ndarray] = None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

class DeepTabularModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class MLModel:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        self.model = DeepTabularModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims
        ).to(device)
        
        self.scaler = StandardScaler()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        self.criterion = nn.MSELoss()
        
    def _create_dataloader(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        shuffle: bool = True
    ) -> DataLoader:
        dataset = TabularDataset(features, targets)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Create dataloaders
        train_loader = self._create_dataloader(X_train_scaled, y_train)
        val_loader = self._create_dataloader(X_val_scaled, y_val, shuffle=False) if X_val is not None else None
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                
                # Model checkpointing
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_checkpoint('best_model.pt')
                
                mlflow.log_metrics({
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        dataloader = self._create_dataloader(X_scaled, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X in dataloader:
                if isinstance(batch_X, tuple):
                    batch_X = batch_X[0]
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions).squeeze()
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'learning_rate': self.learning_rate
            }
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Reconstruct model architecture
        self.input_dim = checkpoint['config']['input_dim']
        self.hidden_dims = checkpoint['config']['hidden_dims']
        self.model = DeepTabularModel(
            self.input_dim,
            self.hidden_dims
        ).to(self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.sum(X, axis=1) + np.random.randn(1000) * 0.1
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Initialize and train model
    model = MLModel(input_dim=10)
    
    with mlflow.start_run():
        mlflow.log_params({
            'input_dim': 10,
            'hidden_dims': model.hidden_dims,
            'learning
