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

def try_read_csv(path: Path):
  # Reading dataset with "tab, comma, whitespace"
    for sep, kwargs in [
        ("\t", {"engine": "python"}),
        (",", {"engine": "python"}),
        (r"\s+", {"engine": "python", "delim_whitespace": True}),
    ]:
        try:
            df = pd.read_csv(path, sep=sep, header=0, **kwargs)
            if df.shape[1] < 3:
                continue
            print(f"خوانده شد با جداکننده: {repr(sep)} — shape: {df.shape}")
            return df
        except Exception as e:
            continue
          raise ValueError(f"نشد فایل را با جداکننده‌های معمول بخوانیم: {path}")

def clean_colnames(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def replace_na_strings(df: pd.DataFrame):
    df = df.copy()
    df = df.replace(["NaN", "nan", "NAN", "", "NULL", "null"], np.nan)
    return df

def str_to_float_safe(s):
    """تبدیل ایمن رشته‌هایی مثل '1,234' یا '-0.1v' یا '7KM' به float.
       اگر تبدیل نشد، None برمی‌گرداند."""
    if s is None:
        return None
    if isinstance(s, (int, float, np.floating, np.integer)):
        return float(s)
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none", "null"]:
        return None
    # برداشتن قسمت‌هایی که مشخصاً غیر عددی اند، مثل 'v'، 'KM', 'km', 'm', '+' یا '±' و غیره
    # اما قبل از حذف علامت منفی/مثبت و ممیز، اعداد با کاما را به نقطه تبدیل کن
    s = s.replace(",", ".")
    # حذف +/±/+/- و پسوندهای متداول
    s = re.sub(r"[\u00B1±\+\/\-]*\s*\+?/?-?\s*\d+(\.\d+)?", "", s)  # remove trailing ±0.10 or +/-0.10
    # حذف واحدهای ساده
    s = re.sub(r"(km|KM|m|M|s|S|v|V|mm|MM|sec|SEC)$", "", s).strip()
    # حالا استخراج اولین عدد اعشاری موجود در رشته
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m.group(0))
        except:
            return None
    return None
