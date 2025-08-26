"""
Data Processing Utilities for Battery SOC Estimation
Stanford TECH 27 Final Project

Shared functions extracted from 02_data_cleaning.ipynb for use across notebooks.
This replaces the old preprocessing_utils.py with corrected validation logic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Global paths
DATA_DIR = Path('dataset/data')
METADATA_FILE = Path('dataset/metadata.csv')
OUTPUT_DIR = Path('processed_data')
OUTPUT_DIR.mkdir(exist_ok=True)

def calculate_soc_from_current(data, initial_soc=1.0, capacity_ah=2.0):
    """
    Calculate State of Charge (SOC) from current integration.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Battery data with Time and Current_measured columns
    initial_soc : float
        Initial state of charge (default: 1.0 for fully charged)
    capacity_ah : float
        Battery capacity in Amp-hours (default: 2.0 Ah)
    
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with SOC column added, or None if insufficient data
    """
    
    # Filter for discharge data (negative current with threshold to avoid noise)
    discharge_mask = data['Current_measured'] < -0.01
    discharge_data = data[discharge_mask].copy()
    
    if len(discharge_data) < 10:  # Need minimum points for meaningful analysis
        return None
    
    # Sort by time to ensure chronological order
    discharge_data = discharge_data.sort_values('Time').reset_index(drop=True)
    
    # Calculate time differences (convert to hours)
    time_diff = discharge_data['Time'].diff().fillna(0) / 3600  # seconds to hours
    
    # Calculate charge consumed (Ah) = Current * time_diff
    # Current is negative during discharge, so negate to get positive charge consumed
    charge_consumed = (-discharge_data['Current_measured'] * time_diff).cumsum()
    
    # Calculate SOC: SOC = initial_SOC - (charge_consumed / capacity)
    soc = initial_soc - (charge_consumed / capacity_ah)
    
    # Add SOC to discharge data
    discharge_data['SOC'] = soc
    
    return discharge_data


def validate_discharge_data(soc_data):
    """
    Validate discharge data quality using multiple criteria.
    
    Parameters:
    -----------
    soc_data : pd.DataFrame
        Discharge data with SOC, Voltage_measured columns
    
    Returns:
    --------
    tuple : (bool, dict)
        (is_valid, validation_details)
    """
    
    if soc_data is None or len(soc_data) < 10:
        return False, {'error': 'insufficient_data'}
    
    soc = soc_data['SOC']
    voltage = soc_data['Voltage_measured']
    
    # Validation Criterion 1: SOC Range Check (0.1 to 1.0)
    # Ensures reasonable discharge range without over-discharge
    soc_min, soc_max = soc.min(), soc.max()
    range_valid = (soc_min >= 0.1) and (soc_max <= 1.0)
    
    # Validation Criterion 2: Voltage-SOC Correlation
    # Strong positive correlation indicates proper voltage-capacity relationship
    correlation = np.corrcoef(voltage, soc)[0, 1] if len(voltage) > 1 else 0
    correlation_valid = (correlation >= 0.7) and not np.isnan(correlation)
    
    # Validation Criterion 3: Monotonic Discharge Trend
    # SOC should generally decrease during discharge (allow small increases due to noise)
    soc_diff = soc.diff().dropna()
    if len(soc_diff) > 0:
        increasing_points = (soc_diff > 0.02).sum()  # Significant increases
        increasing_percentage = (increasing_points / len(soc_diff)) * 100
        trend_valid = increasing_percentage < 20  # Less than 20% increasing
    else:
        trend_valid = False
        increasing_percentage = 100
    
    # Overall validation result
    all_valid = range_valid and correlation_valid and trend_valid
    
    validation_details = {
        'range_valid': range_valid,
        'correlation_valid': correlation_valid,
        'trend_valid': trend_valid,
        'soc_range': (soc_min, soc_max),
        'correlation': correlation,
        'increasing_percentage': increasing_percentage
    }
    
    return all_valid, validation_details


def process_single_discharge_file(file_path):
    """
    Process a single discharge file with SOC calculation and validation.
    
    Parameters:
    -----------
    file_path : Path
        Path to the CSV file
    
    Returns:
    --------
    tuple : (bool, pd.DataFrame or str)
        (success, data_or_error_message)
    """
    
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Check required columns
        required_columns = ['Current_measured', 'Voltage_measured', 'Time']
        if not all(col in data.columns for col in required_columns):
            return False, f"Missing required columns: {required_columns}"
        
        # Calculate SOC
        soc_data = calculate_soc_from_current(data)
        
        if soc_data is None:
            return False, "Insufficient discharge data points"
        
        # Validate data quality
        is_valid, validation_details = validate_discharge_data(soc_data)
        
        if not is_valid:
            reasons = []
            if not validation_details.get('range_valid', True):
                soc_min, soc_max = validation_details['soc_range']
                reasons.append(f"SOC range ({soc_min:.3f} to {soc_max:.3f})")
            if not validation_details.get('correlation_valid', True):
                corr = validation_details.get('correlation', 0)
                reasons.append(f"Low correlation ({corr:.3f})")
            if not validation_details.get('trend_valid', True):
                pct = validation_details.get('increasing_percentage', 0)
                reasons.append(f"Non-monotonic ({pct:.1f}% increasing)")
            
            return False, f"Validation failed: {', '.join(reasons)}"
        
        return True, soc_data
        
    except Exception as e:
        return False, f"Processing error: {str(e)}"


def load_cleaned_data(cleaned_data_path=None):
    """
    Load cleaned discharge data from 02_data_cleaning.ipynb output.
    
    Parameters:
    -----------
    cleaned_data_path : Path or None
        Path to cleaned data CSV file. If None, uses default location.
    
    Returns:
    --------
    pd.DataFrame
        Cleaned discharge data with SOC column
    """
    
    if cleaned_data_path is None:
        cleaned_data_path = OUTPUT_DIR / 'cleaned_discharge_data.csv'
    
    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {cleaned_data_path}")
    
    data = pd.read_csv(cleaned_data_path)
    print(f"Loaded cleaned data: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Unique batteries: {data['battery_id'].nunique()}")
    print(f"SOC range: {data['SOC'].min():.3f} to {data['SOC'].max():.3f}")
    
    return data


def engineer_features(data):
    """
    Engineer comprehensive features from cleaned battery discharge data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Cleaned discharge data with SOC, Voltage_measured, Current_measured, etc.
    
    Returns:
    --------
    pd.DataFrame
        Data with engineered features
    """
    
    # Start with a copy to avoid modifying original data
    features = data.copy()
    
    # Rename columns for consistency
    features = features.rename(columns={
        'Voltage_measured': 'voltage',
        'Current_measured': 'current',
        'Temperature_measured': 'temperature',
        'Time': 'time'
    })
    
    # Basic derived features
    features['power'] = features['voltage'] * abs(features['current'])
    features['abs_current'] = abs(features['current'])
    features['voltage_current_ratio'] = features['voltage'] / features['abs_current']
    features['energy'] = features['power'] * features['time'].diff().fillna(0) / 3600  # Wh
    
    # Rate of change features
    features['voltage_change'] = features['voltage'].diff()
    features['current_change'] = features['current'].diff() 
    features['temperature_change'] = features['temperature'].diff()
    features['power_change'] = features['power'].diff()
    
    # Rolling statistics (multiple windows)
    for window in [5, 10, 20]:
        features[f'voltage_rolling_mean_{window}'] = features['voltage'].rolling(window, min_periods=1).mean()
        features[f'current_rolling_mean_{window}'] = features['current'].rolling(window, min_periods=1).mean()
        features[f'temperature_rolling_mean_{window}'] = features['temperature'].rolling(window, min_periods=1).mean()
        features[f'voltage_rolling_std_{window}'] = features['voltage'].rolling(window, min_periods=1).std()
        features[f'current_rolling_std_{window}'] = features['current'].rolling(window, min_periods=1).std()
    
    # Cumulative features
    features['cumulative_discharge'] = features['abs_current'].cumsum() * features['time'].diff().fillna(0) / 3600
    features['cumulative_energy'] = features['energy'].cumsum()
    
    # Time-based features
    features['time_normalized'] = (features['time'] - features['time'].min()) / (features['time'].max() - features['time'].min())
    features['time_since_start'] = features['time'] - features['time'].min()
    
    # Lag features (previous values)
    for lag in [1, 3, 5]:
        features[f'voltage_lag_{lag}'] = features['voltage'].shift(lag)
        features[f'current_lag_{lag}'] = features['current'].shift(lag)
        features[f'SOC_lag_{lag}'] = features['SOC'].shift(lag)
    
    # Statistical features (position relative to sequence statistics)
    features['voltage_vs_mean'] = features['voltage'] - features['voltage'].mean()
    features['current_vs_mean'] = features['current'] - features['current'].mean()
    features['voltage_percentile'] = features['voltage'].rank(pct=True)
    features['current_percentile'] = features['current'].rank(pct=True)
    
    # Fill NaN values created by rolling/lag operations
    features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return features


def prepare_sequences(data, sequence_length=30, step=5, target_col='SOC'):
    """
    Prepare sequential data for RNN/LSTM models.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with features and target column
    sequence_length : int
        Number of timesteps in each sequence
    step : int
        Step size between sequences
    target_col : str
        Target column name
    
    Returns:
    --------
    tuple : (X, y, feature_names)
        X: 3D array (samples, timesteps, features)
        y: 1D array of targets
        feature_names: list of feature names
    """
    
    # Select feature columns (exclude metadata and target)
    exclude_cols = ['SOC', 'battery_id', 'filename', 'test_id', 'ambient_temperature', 'time']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        return np.array([]), np.array([]), []
    
    # Get features and target
    X_data = data[feature_cols].values
    y_data = data[target_col].values
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(0, len(X_data) - sequence_length + 1, step):
        X_sequences.append(X_data[i:i + sequence_length])
        y_sequences.append(y_data[i + sequence_length - 1])  # Predict last value in sequence
    
    if len(X_sequences) == 0:
        return np.array([]), np.array([]), feature_cols
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    return X, y, feature_cols


def prepare_cnn_sequences(data, sequence_length=100, step=10, target_col='SOC'):
    """
    Prepare longer sequential data for 1D CNN models.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with features and target column
    sequence_length : int
        Number of timesteps in each sequence (longer for CNNs)
    step : int
        Step size between sequences
    target_col : str
        Target column name
    
    Returns:
    --------
    tuple : (X, y, feature_names)
        X: 3D array (samples, timesteps, features)
        y: 1D array of targets
        feature_names: list of feature names
    """
    
    # Use same logic as LSTM but with different default parameters
    return prepare_sequences(data, sequence_length, step, target_col)


def create_train_val_test_splits(data, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create train/validation/test splits based on battery IDs to prevent data leakage.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with battery_id column
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set (from remaining after test split)
    random_state : int
        Random seed
    
    Returns:
    --------
    dict
        Dictionary with 'train', 'val', 'test' dataframes
    """
    
    # Get unique batteries
    unique_batteries = data['battery_id'].unique()
    
    # Split batteries into train/val/test
    train_val_batteries, test_batteries = train_test_split(
        unique_batteries, test_size=test_size, random_state=random_state
    )
    
    train_batteries, val_batteries = train_test_split(
        train_val_batteries, test_size=val_size/(1-test_size), random_state=random_state
    )
    
    # Split data based on battery assignment
    train_data = data[data['battery_id'].isin(train_batteries)]
    val_data = data[data['battery_id'].isin(val_batteries)]
    test_data = data[data['battery_id'].isin(test_batteries)]
    
    print(f"Battery-based splits:")
    print(f"  Train: {len(train_batteries)} batteries, {len(train_data)} samples")
    print(f"  Val:   {len(val_batteries)} batteries, {len(val_data)} samples")  
    print(f"  Test:  {len(test_batteries)} batteries, {len(test_data)} samples")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


def prepare_ml_datasets(splits, target_col='SOC', scale_features=True):
    """
    Prepare datasets for conventional ML models with proper scaling.
    
    Parameters:
    -----------
    splits : dict
        Dictionary with 'train', 'val', 'test' dataframes
    target_col : str
        Target column name
    scale_features : bool
        Whether to apply StandardScaler to features
    
    Returns:
    --------
    dict
        Dictionary with scaled X/y datasets and metadata
    """
    
    # Select feature columns
    exclude_cols = ['SOC', 'battery_id', 'filename', 'test_id', 'ambient_temperature', 'time']
    feature_cols = [col for col in splits['train'].columns if col not in exclude_cols]
    
    # Extract features and targets
    X_train = splits['train'][feature_cols]
    X_val = splits['val'][feature_cols]
    X_test = splits['test'][feature_cols]
    
    y_train = splits['train'][target_col]
    y_val = splits['val'][target_col]
    y_test = splits['test'][target_col]
    
    # Scale features if requested
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    
    return {
        'X_train': X_train,
        'X_val': X_val, 
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'scaler': scaler
    }


def prepare_sequence_ml_datasets(data_splits, sequence_length=30, step=5, target_col='SOC'):
    """
    Prepare sequence datasets for deep learning models with proper splits.
    
    Parameters:
    -----------
    data_splits : dict
        Dictionary with 'train', 'val', 'test' dataframes
    sequence_length : int
        Sequence length for models
    step : int
        Step size between sequences
    target_col : str
        Target column name
    
    Returns:
    --------
    dict
        Dictionary with sequence datasets
    """
    
    sequences = {}
    
    for split_name, split_data in data_splits.items():
        X, y, feature_names = prepare_sequences(split_data, sequence_length, step, target_col)
        sequences[f'X_{split_name}'] = X
        sequences[f'y_{split_name}'] = y
    
    sequences['feature_names'] = feature_names
    
    # Print shapes
    if len(sequences['X_train']) > 0:
        print(f"Sequence dataset shapes:")
        for split in ['train', 'val', 'test']:
            X_key, y_key = f'X_{split}', f'y_{split}'
            if X_key in sequences:
                print(f"  {split}: X={sequences[X_key].shape}, y={sequences[y_key].shape}")
    
    return sequences


def save_datasets(ml_datasets, sequence_datasets=None, output_prefix="processed"):
    """
    Save processed datasets to pickle files.
    
    Parameters:
    -----------
    ml_datasets : dict
        ML datasets from prepare_ml_datasets
    sequence_datasets : dict or None
        Sequence datasets from prepare_sequence_ml_datasets
    output_prefix : str
        Prefix for output filenames
    """
    
    # Save ML datasets
    ml_path = OUTPUT_DIR / f'{output_prefix}_ml_datasets.pkl'
    with open(ml_path, 'wb') as f:
        pickle.dump(ml_datasets, f)
    print(f"Saved ML datasets to {ml_path}")
    
    # Save sequence datasets if provided
    if sequence_datasets:
        seq_path = OUTPUT_DIR / f'{output_prefix}_sequence_datasets.pkl'
        with open(seq_path, 'wb') as f:
            pickle.dump(sequence_datasets, f)
        print(f"Saved sequence datasets to {seq_path}")
        return {'ml_path': ml_path, 'sequence_path': seq_path}
    
    return {'ml_path': ml_path, 'sequence_path': None}