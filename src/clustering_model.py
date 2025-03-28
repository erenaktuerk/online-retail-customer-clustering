from sklearn.cluster import KMeans
import pandas as pd

class ClusteringModel:
    def __init__(self, n_clusters=3):
        """
        Initialize the clustering model with the number of clusters.
        Args:
            n_clusters (int): Number of clusters for KMeans.
        """
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)

    def fit(self, df):
        """
        Fit the clustering model on the data.
        Args:
            df (pd.DataFrame): Data to be clustered.
        """
        self.model.fit(df)
        # Cluster-Zugehörigkeit zu den Daten hinzufügen
        df['Cluster'] = self.model.labels_
        
        # Cluster-Profile berechnen (Durchschnittswerte je Cluster)
        cluster_profiles = df.groupby('Cluster').mean()
        
        # Cluster-Profile ausgeben
        print("\nCluster Profiles (Durchschnittswerte je Cluster):")
        print(cluster_profiles)
        
        return df

    def predict(self, df):
        """
        Predict the cluster labels for the data.
        Args:
            df (pd.DataFrame): Data to predict the clusters.
        Returns:
            cluster_labels (array): Predicted cluster labels.
        """
        return self.model.predict(df)