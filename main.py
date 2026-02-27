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
from scipy.spatial import cKDTree, gammaln, Delaunay
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

# input and output path
INPUT_PATH = Path("final_data.csv")
OUTPUT_PATH = Path("final_data_cleaned.csv")

# User setting
mag_col = "MAG"              
min_magnitude = 4.0          

# Reading csv file function
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
    if s is None:
        return None
    if isinstance(s, (int, float, np.floating, np.integer)):
        return float(s)
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none", "null"]:
        return None
    s = s.replace(",", ".")
    s = re.sub(r"[\u00B1±\+\/\-]*\s*\+?/?-?\s*\d+(\.\d+)?", "", s)  # remove trailing ±0.10 or +/-0.10
    s = re.sub(r"(km|KM|m|M|s|S|v|V|mm|MM|sec|SEC)$", "", s).strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m.group(0))
        except:
            return None
    return None

def col_to_numeric(df: pd.DataFrame, colnames):
    for col in colnames:
        if col not in df.columns:
            continue
        df[col + "_raw"] = df[col]  
        df[col] = df[col].apply(str_to_float_safe)
    return df

def parse_event_time(s):
    """
       '2019Y  6M 21D  0H  0M 16.05S +/-0.10 EASTERN F...'      
    """
    if pd.isna(s):
        return None
    s = str(s)
    # year Y, month M, day D, hour H, minute M, second S
    m = re.search(
        r"(?P<Y>\d{4})\s*Y.*?(?P<M>\d{1,2})\s*M.*?(?P<D>\d{1,2})\s*D.*?(?P<h>\d{1,2})\s*H.*?(?P<m>\d{1,2})\s*M.*?(?P<s>\d{1,2}(?:\.\d+)?)\s*S",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        nums = re.findall(r"\d{1,4}\.\d+|\d{1,4}", s)
        if len(nums) >= 6:
            try:
                Y, M, D, h, mm, ss = nums[:6]
                return datetime(
                    int(Y), int(M), int(D), int(h), int(mm), int(float(ss))
                )
            except Exception:
                return None
        return None
    try:
        Y = int(m.group("Y"))
        M = int(m.group("M"))
        D = int(m.group("D"))
        h = int(m.group("h"))
        mm = int(m.group("m"))
        ss = float(m.group("s"))
        sec = int(ss)
        micro = int((ss - sec) * 1_000_000)
        return datetime(Y, M, D, h, mm, sec, micro)
    except Exception:
        return None

