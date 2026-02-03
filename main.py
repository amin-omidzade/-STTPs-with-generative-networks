# Omidzadehnik
# Jan 22 2026

# Import Modules
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Reading dataset
data = pd.read_csv(data.csv)

# ===========================
# 1. Data Preprocessing
# ===========================

class STPPDataPreprocessor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.scaler = MinMaxScaler()
    
    def clean_data(self, min_magnitude=3.0, min_quality=0.8, spatial_bounds=None):
        # Scale Filtering
        cleaned = self.data[self.data['magnitude'] >= min_magnitude]
        # Quality Filtering
        cleaned = cleaned[cleaned['quality'] >= min_quality]
        # Spatial Filtering
        if spatial_bounds:
            xmin, xmax, ymin, ymax = spatial_bounds
            cleaned = cleaned[
                (cleaned['longitude'] >= xmin) & (cleaned['longitude'] <= xmax) &
                (cleaned['latitude'] >= ymin) & (cleaned['latitude'] <= ymax)
            ]
        # Remove duplicates
        cleaned = cleaned.drop_duplicates(subset=['time'])
        self.cleaned_data = cleaned.reset_index(drop=True)
    
    def normalize(self):
        coords = self.cleaned_data[['longitude', 'latitude']].values
        times = self.cleaned_data[['time']].values
        coords_norm = self.scaler.fit_transform(coords)
        times_norm = self.scaler.fit_transform(times)
        self.normalized_data = np.hstack([coords_norm, times_norm])
    
    def create_density_maps(self, grid_size=50, bandwidth=0.05):
        # Grid STTPs network
        x_grid = np.linspace(0, 1, grid_size)
        y_grid = np.linspace(0, 1, grid_size)
        t_grid = np.linspace(0, 1, grid_size)
        X, Y, T = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')
        grid_points = np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T
        
        # Calculation of Guassian kernel
        density = np.zeros(grid_points.shape[0])
        for event in self.normalized_data:
            dist = np.linalg.norm(grid_points - event, axis=1)
            density += np.exp(-dist**2 / (2 * bandwidth**2))
        
        self.density_map = density.reshape(X.shape)
        return self.density_map
    
    def split_train_test(self, test_ratio=0.2):
        n = len(self.normalized_data)
        split_idx = int(n * (1 - test_ratio))
        train_data = self.normalized_data[:split_idx]
        test_data = self.normalized_data[split_idx:]
        return train_data, test_data

