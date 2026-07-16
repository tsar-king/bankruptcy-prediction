"""
Data loading and splitting utilities.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str = 'data/raw/data.csv') -> pd.DataFrame:
    """
    Load the Taiwanese Bankruptcy dataset and standardize column names.
    
    Args:
        path: Path to the CSV file.
    
    Returns:
        DataFrame with standardized column names.
    """
    df = pd.read_csv(path)
    
    # Standardize target column name (works for UCI, Kaggle, OpenML)
    target_candidates = ['Bankrupt?', 'Bankrupt', 'class', 'target']
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        target_col = df.columns[0]  # assume first column
    
    # Rename target to 'Bankrupt'
    df = df.rename(columns={target_col: 'Bankrupt'})
    
    # Rename feature columns to X1...X95 if not already
    feature_cols = [c for c in df.columns if c != 'Bankrupt']
    if len(feature_cols) == 95:
        rename_map = {old: f'X{i+1}' for i, old in enumerate(feature_cols)}
        df = df.rename(columns=rename_map)
    
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]-1} features")
    print(f"Bankruptcy rate: {df['Bankrupt'].mean()*100:.2f}%")
    
    return df


def split_data(df: pd.DataFrame, target_col: str = 'Bankrupt', 
               test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and hold-out test sets with stratification.
    
    Args:
        df: Full DataFrame.
        target_col: Name of target column.
        test_size: Proportion for test set.
        random_state: Seed for reproducibility.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples ({y_train.mean()*100:.2f}% bankrupt)")
    print(f"Test set:  {X_test.shape[0]} samples ({y_test.mean()*100:.2f}% bankrupt)")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick test
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)