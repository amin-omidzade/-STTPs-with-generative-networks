# Omidzadehnik
# Jan 22 2026

# Import Modules
import re
from pathlib import Path
import os
import warnings

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import json
import random

from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import gammaln
from scipy.spatial import Delaunay

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

from shapely import wkt as shapely_wkt
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# in and out path
INPUT_PATH = Path("final_data.csv")
OUTPUT_PATH = Path("final_data_cleaned.csv")

# User setting
mag_col = "MAG"              
min_magnitude = 4.0          


