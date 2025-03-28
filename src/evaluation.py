"""
Module for model evaluation.
Provides the Evaluation class to calculate clustering performance metrics.
"""

from sklearn.metrics import silhouette_score

class Evaluation:
    def calculate_silhouette(self, X, labels):
        """
        Calculate the Silhouette Score for the clustering.
        Args:
            X (pd.DataFrame or array-like): Data used for clustering.
            labels (array): Cluster labels.
        Returns:
            score (float): Silhouette Score.
        """
        score = silhouette_score(X, labels)
        return score