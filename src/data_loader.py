"""
Data loading module for Bank Fraud Detection project.
"""
import pandas as pd
from pathlib import Path
from src.config import DATA_FILE, RAW_DATA_DIR


class DataLoader:
    """Handles loading and basic validation of the fraud detection dataset."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV file. If None, uses default from config.
        """
        self.data_path = Path(data_path) if data_path else DATA_FILE
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the credit card fraud detection dataset.
        
        Returns:
            DataFrame with the loaded data.
            
        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {self.data_path}. "
                f"Please download the Credit Card Fraud Detection dataset from Kaggle "
                f"and place it at {RAW_DATA_DIR}/creditcard.csv"
            )
        
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Dictionary with dataset information.
        """
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "class_distribution": df["Class"].value_counts().to_dict() if "Class" in df.columns else None,
            "fraud_percentage": (df["Class"].sum() / len(df) * 100) if "Class" in df.columns else None
        }
        return info
    
    def print_data_info(self, df: pd.DataFrame):
        """Print formatted data information."""
        info = self.get_data_info(df)
        print("\n=== Dataset Information ===")
        print(f"Shape: {info['shape']}")
        print(f"Columns: {len(info['columns'])}")
        print(f"\nClass Distribution: {info['class_distribution']}")
        print(f"Fraud Percentage: {info['fraud_percentage']:.4f}%")
        print(f"\nMissing Values: {sum(info['missing_values'].values())}")
        print(f"Data Types:\n{pd.Series(info['dtypes']).value_counts()}")


def download_kaggle_dataset():
    """
    Instructions for downloading the dataset from Kaggle.
    This function provides guidance since automated download requires Kaggle API setup.
    """
    print("""
To download the Credit Card Fraud Detection dataset:

1. Install Kaggle API:
   pip install kaggle

2. Set up Kaggle credentials:
   - Go to https://www.kaggle.com/account
   - Click 'Create New API Token' (downloads kaggle.json)
   - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)

3. Download dataset:
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip

Or manually:
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download creditcard.csv
3. Place it in: data/raw/creditcard.csv
""")