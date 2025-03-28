"""
Module for data preprocessing.
Provides the DataPreprocessor class to load, clean, and transform data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, data_path):
        """
        Initialize the DataPreprocessor with the path to the data file.
        """
        self.data_path = data_path
        self.scaler = StandardScaler()

    def load_data(self, chunk_size=50000):
        """
        Load data from a CSV file in chunks to handle large files.
        Args:
            chunk_size (int): Number of rows per chunk to be loaded.
        Returns:
            chunks (generator): Generator that yields chunks of data.
        """
        # Load the CSV file in chunks
        chunks = pd.read_csv(self.data_path, encoding='ISO-8859-1', chunksize=chunk_size)
        
        # Return the generator that yields chunks
        return chunks

    def clean_data(self, df):
        """
        Clean the data by handling missing values and removing irrelevant columns.
        Args:
            df (pd.DataFrame): Raw data.
        Returns:
            df_clean (pd.DataFrame): Cleaned data.
        """
        # Example: Remove rows with missing values
        df_clean = df.dropna()
        # Example: Keep only numeric columns for clustering
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        df_clean = df_clean[numeric_cols]
        return df_clean

    def transform_data(self, df):
        """
        Transform data using scaling.
        Args:
            df (pd.DataFrame): Cleaned data.
        Returns:
            df_transformed (pd.DataFrame): Scaled data.
        """
        scaled_array = self.scaler.fit_transform(df)
        df_transformed = pd.DataFrame(scaled_array, columns=df.columns)
        return df_transformed