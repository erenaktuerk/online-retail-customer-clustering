"""
Unit tests for the clustering_model module.
"""

import unittest
import pandas as pd
from src.clustering_model import ClusteringModel

class TestClusteringModel(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset
        self.data = pd.DataFrame({
            'feature1': [1, 2, 1, 2],
            'feature2': [1, 1, 2, 2]
        })
        self.model = ClusteringModel(n_clusters=2)

    def test_fit_predict(self):
        self.model.fit(self.data)
        labels = self.model.predict(self.data)
        # Check that there are as many labels as data points
        self.assertEqual(len(labels), len(self.data))

if __name__ == '__main__':
    unittest.main()