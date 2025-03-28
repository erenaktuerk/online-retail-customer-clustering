"""
Unit tests for the visualization module.
"""

import unittest
import pandas as pd
import os
from src.visualization import Plotter

class TestPlotter(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset
        self.data = pd.DataFrame({
            'feature1': [1, 2, 1, 2],
            'feature2': [1, 1, 2, 2]
        })
        self.labels = [0, 1, 0, 1]
        self.plotter = Plotter()

    def test_plot_clusters(self):
        output_path = 'results/figures/test_clusters.png'
        # Ensure the directory exists
        os.makedirs('results/figures', exist_ok=True)
        self.plotter.plot_clusters(self.data, self.labels, output_path=output_path)
        # Check if the plot file was created
        self.assertTrue(os.path.exists(output_path))

if __name__ == '__main__':
    unittest.main()