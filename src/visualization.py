"""
Module for visualization.
Provides the Plotter class to visualize clustering results.
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Plotter:
    def plot_clusters(self, df, labels, output_path='clusters.png'):
        """
        Visualize clusters using PCA for dimensionality reduction.
        Args:
            df (pd.DataFrame): Data used for clustering.
            labels (array): Cluster labels.
            output_path (str): File path to save the plot.
        """
        pca = PCA(n_components=2)
        components = pca.fit_transform(df)
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(components[:,0], components[:,1], c=labels, cmap='viridis', alpha=0.7)
        plt.title('Cluster Visualization using PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Cluster Label')
        plt.savefig(output_path)
        plt.close()
        