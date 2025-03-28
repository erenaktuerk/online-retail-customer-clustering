from src.data_preprocessing import DataPreprocessor
from src.clustering_model import ClusteringModel
from src.evaluation import Evaluation
from src.visualization import Plotter
import pandas as pd

def main():
    # Step 1: Data Loading and Preprocessing
    data_path = 'data/raw/OnlineRetail.csv'  # Adjust filename as necessary
    preprocessor = DataPreprocessor(data_path)
    df = preprocessor.load_data()
    df_clean = preprocessor.clean_data(df)
    df_processed = preprocessor.transform_data(df_clean)
    
    # Save processed data
    df_processed.to_csv('data/processed/processed_data.csv', index=False)
    
    # Step 2: Clustering
    clustering = ClusteringModel(n_clusters=3)
    clustering.fit(df_processed)
    cluster_labels = clustering.predict(df_processed)
    df_processed['Cluster'] = cluster_labels
    
    # Save clustering output
    df_processed.to_csv('results/clustering_output.csv', index=False)
    
    # Step 3: Evaluation
    evaluator = Evaluation()
    score = evaluator.calculate_silhouette(df_processed.drop('Cluster', axis=1), cluster_labels)
    print(f"Silhouette Score: {score:.3f}")
    
    # Step 4: Visualization
    plotter = Plotter()
    plotter.plot_clusters(df_processed.drop('Cluster', axis=1), cluster_labels, output_path='results/figures/clusters.png')

if __name__ == '__main__':
    main()