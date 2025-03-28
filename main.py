from src.data_preprocessing import DataPreprocessor
from src.clustering_model import ClusteringModel
from src.evaluation import Evaluation
from src.visualization import Plotter
import pandas as pd

def main():
    # Step 1: Data Loading and Preprocessing
    data_path = 'data/raw/OnlineRetail.csv'
    preprocessor = DataPreprocessor(data_path)

    # Load data in chunks
    chunks = preprocessor.load_data()
    
    # Process each chunk
    for chunk in chunks:
        df_clean = preprocessor.clean_data(chunk)
        df_processed = preprocessor.transform_data(df_clean)
        
        # Weiterverarbeiten, Speichern oder Clustering für jedes Chunk
        # z.B. Data speichern oder aggregieren
        
    # Step 2: Clustering and further processing (as in the original code)
    clustering = ClusteringModel(n_clusters=3)
    df_with_clusters = clustering.fit(df_processed)  # Gibt df_with_clusters mit Cluster-Spalte zurück
    
    # Hier wird das Cluster-Profil direkt nach dem Clustering angezeigt
    # (bereits in der fit-Methode des ClusteringModel integriert)
    
    # Step 3: Evaluation
    cluster_labels = df_with_clusters['Cluster'].values  # Cluster Labels holen
    evaluator = Evaluation()
    score = evaluator.calculate_silhouette(df_processed.drop('Cluster', axis=1), cluster_labels)
    print(f"Silhouette Score: {score:.3f}")
    
    # Step 4: Visualization
    plotter = Plotter()
    plotter.plot_clusters(df_processed.drop('Cluster', axis=1), cluster_labels, output_path='results/figures/clusters.png')

if __name__ == '__main__':
    main()