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

