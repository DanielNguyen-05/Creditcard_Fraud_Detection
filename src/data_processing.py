import numpy as np
import os

def load_csv_numpy(path, delimiter=',', skip_header=1):
    """
    Load CSV to numpy array (float64) handling quoted numbers.
    
    Args:
        path (str): Path to the CSV file.
        delimiter (str): Separator used in CSV.
        skip_header (int): Number of header rows to skip.
        
    Returns:
        tuple: (header_list, data_array)
    """
    # 1. Read header separately to handle potential quotes in column names
    with open(path, 'r', encoding='utf-8-sig') as f:
        header_line = f.readline().strip()
        # Remove quotes and split
        header = header_line.replace('"', '').split(delimiter)
    
    # 2. Load data as String first to avoid NaN errors caused by quotes (e.g., "0.12")
    try:
        data_str = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header, dtype=str)
        
        # 3. Remove quotation marks globally
        data_clean = np.char.replace(data_str, '"', '')
        
        # 4. Convert to Float
        data = data_clean.astype(float)
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return header, np.array([])

    return header, data

def save_processed(folder, **arrays):
    """
    Save multiple numpy arrays into a compressed .npz file.
    """
    os.makedirs(folder, exist_ok=True)
    savepath = os.path.join(folder, 'processed.npz')
    np.savez_compressed(savepath, **arrays)
    return savepath

def check_missing(X):
    """Checks for NaN values in the array."""
    return np.isnan(X).any(), np.isnan(X).sum()

def impute_missing(X, strategy='median'):
    """
    Fill NaN values using Mean or Median of the column.
    """
    X2 = X.copy()
    mask = np.isnan(X2)
    if not mask.any():
        return X2
        
    # Iterate over columns (permitted as n_features is small)
    for col in range(X2.shape[1]):
        col_mask = mask[:, col]
        if not col_mask.any():
            continue
        
        if strategy == 'mean':
            val = np.nanmean(X2[:, col])
        else:
            val = np.nanmedian(X2[:, col])
            
        X2[col_mask, col] = val
    return X2

def zscore(X, eps=1e-12):
    """
    Standardization: (X - mean) / std.
    Returns the transformed array AND stats (to avoid data leakage later).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Avoid division by zero
    std_adj = np.where(std < eps, 1.0, std)
    Xs = (X - mean) / std_adj
    return Xs, mean, std_adj

def minmax_scale(X, eps=1e-12):
    """
    Min-Max Scaling: (X - min) / (max - min).
    """
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    rng = mx - mn
    rng_adj = np.where(rng < eps, 1.0, rng)
    Xs = (X - mn) / rng_adj
    return Xs, mn, mx

def extract_time_features(time_col):
    """
    Converts raw seconds (0-172792) into Hour of Day (0-23).
    """
    seconds = time_col.astype(np.int64)
    hour = (seconds // 3600) % 24
    return hour.reshape(-1, 1)

def stratified_split(X, y, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Splits data into Train/Val/Test while maintaining class ratios.
    This is critical for Imbalanced Datasets.
    """
    # Sanity check
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []
    
    for c in classes:
        # Get indices for this class
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        
        # Calculate split sizes
        n_train = int(np.floor(train_frac * n))
        n_val = int(np.floor(val_frac * n))
        
        # Slice indices
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])
    
    # Concatenate and shuffle final indices
    train_idx = np.concatenate(train_idx).astype(int)
    val_idx = np.concatenate(val_idx).astype(int)
    test_idx = np.concatenate(test_idx).astype(int)
    
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])

def simple_oversample(X, y, seed=0):
    """
    Random Oversampling: Duplicates minority class samples until balanced.
    Use this ONLY on Training data.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    
    Xs, ys = [X.copy()], [y.copy()]
    
    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) == 0: continue
            
        n_needed = max_count - len(idx)
        if n_needed <= 0: continue
            
        # Randomly select existing samples to duplicate
        reps = rng.choice(idx, size=n_needed, replace=True)
        Xs.append(X[reps])
        ys.append(y[reps])
        
    X_new = np.vstack(Xs)
    y_new = np.concatenate(ys)
    
    # Final shuffle
    perm = rng.permutation(len(y_new))
    return X_new[perm], y_new[perm]

def calculate_t_test(X, y, feature_idx):
    """
    Calculates T-statistic for a specific feature between Class 0 and Class 1.
    Formula: t = (mean1 - mean2) / sqrt(s1^2/n1 + s2^2/n2)
    """
    group0 = X[y == 0, feature_idx]
    group1 = X[y == 1, feature_idx]
    
    n0, n1 = len(group0), len(group1)
    if n0 < 2 or n1 < 2: return 0.0 # Not enough data
    
    mean0, mean1 = np.mean(group0), np.mean(group1)
    std0, std1 = np.std(group0, ddof=1), np.std(group1, ddof=1)
    
    # Standard Error Calculation
    se = np.sqrt((std0**2 / n0) + (std1**2 / n1))
    
    # T-statistic
    t_stat = (mean0 - mean1) / (se + 1e-12)
    return t_stat


def get_subset_from_split(X, y, n_samples=5000, fraud_ratio=None):
    """
    The function retrieves a subset from the existing dataset.
    - If fraud_ratio=None: Keep the original ratio (Random Choice).
    - If fraud_ratio=0.x: Attempt to rebalance the Fraud ratio (Stratified Sampling).
    """
    idx_fraud = np.where(y == 1)[0]
    idx_normal = np.where(y == 0)[0]
    
    if fraud_ratio is None:
        indices = np.random.choice(len(y), n_samples, replace=False)
        return X[indices], y[indices]
    
    n_fraud_needed = int(n_samples * fraud_ratio)
    n_normal_needed = n_samples - n_fraud_needed
    
    actual_fraud = min(n_fraud_needed, len(idx_fraud))
    actual_normal = min(n_normal_needed, len(idx_normal))
    
    idx_f_sub = np.random.choice(idx_fraud, actual_fraud, replace=False)
    idx_n_sub = np.random.choice(idx_normal, actual_normal, replace=False)
    
    subset_idx = np.concatenate([idx_f_sub, idx_n_sub])
    np.random.shuffle(subset_idx) 
    
    return X[subset_idx], y[subset_idx]