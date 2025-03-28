"""
Module for clustering models.
Provides the ClusteringModel class to perform clustering using KMeans.
"""

from sklearn.cluster import KMeans

class ClusteringModel:
    def _init_(self, n_clusters=3, random_state=42):
        """
        Initialize the ClusteringModel with the specified number of clusters.
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X):
        """
        Fit the clustering model.
        Args:
            X (pd.DataFrame or array-like): The data to fit.
        """
        self.model.fit(X)

    def predict(self, X):
        """
        Predict cluster labels for the given data.
        Args:
            X (pd.DataFrame or array-like): Data for prediction.
        Returns:
            labels (array): Predicted cluster labels.
        """
        return self.model.predict(X)