"""
Unit tests for the data_preprocessing module.
"""

import unittest
import pandas as pd
from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a simple DataFrame for testing
        data = {
            'A': [1, 2, None, 4],
            'B': [5, 6, 7, 8],
            'C': ['x', 'y', 'z', 'w']
        }
        self.df = pd.DataFrame(data)
        self.preprocessor = DataPreprocessor(None)

    def test_clean_data(self):
        df_clean = self.preprocessor.clean_data(self.df)
        # Check that there are no missing values
        self.assertFalse(df_clean.isnull().values.any())
        # Check that only numeric columns are retained
        self.assertListEqual(list(df_clean.columns), ['A', 'B'])

if __name__ == '__main__':
    unittest.main()