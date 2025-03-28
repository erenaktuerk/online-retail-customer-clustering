# Unsupervised Learning (Clustering) Project

## Overview
This project demonstrates a complete pipeline for an unsupervised learning task (clustering) using a real-world dataset from Kaggle. The pipeline includes data loading, preprocessing, clustering with KMeans, evaluation using the Silhouette Score, and visualization with PCA.

## Project Structure
- *data/*  
  - raw/: Contains raw data downloaded directly from Kaggle.
  - processed/: Contains processed data after cleaning and transformation.
  - external/: Contains any external datasets if needed.

- *notebooks/*  
  Jupyter notebooks for exploratory data analysis, preprocessing, clustering, and visualization.
  
- *src/*
  Contains the source code for the project:
  - *data_preprocessing.py*: Data loading, cleaning, and transformation.
  - *clustering_model.py*: Implementation of clustering algorithms.
  - *visualization.py*: Visualization of clustering results, including PCA for dimensionality reduction.
  - *evaluation.py*: Evaluation of clustering performance, including the calculation of the Silhouette Score.
  
- *results/*  
  Stores the output files, including visualizations and CSV files with clustering results and evaluation metrics.
  
- *models/*  
  Stores trained/saved models (e.g., a pickled clustering model).

- *tests/*  
  Unit tests for the modules in the src/ folder.

- *main.py*  
  The main entry point that orchestrates the complete pipeline, including data loading, preprocessing, clustering, evaluation, and visualization.

- *.gitignore*  
  Specifies files and directories to be ignored by Git (including sensitive data like the Kaggle API key).

- *requirements.txt*  
  List of required Python packages.

- *LICENSE*  
  The project license (MIT License).

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/erenaktuerk/Unsupervised_Learning_Project.git
   cd Unsupervised_Learning_Project

	2.	Create and activate a virtual environment:

python -m venv venv
.\venv\Scripts\activate  # On Windows


	3.	Install dependencies:

pip install -r requirements.txt


	4.	Kaggle API Setup:
	•	Download your kaggle.json from your Kaggle account.
	•	Create a folder named .kaggle in your user directory (e.g., C:\Users<YourUser>.kaggle) and place the kaggle.json file there.
	•	Ensure that .kaggle/ is added to your .gitignore to avoid committing sensitive information.
	5.	Download the Dataset:
Use the Kaggle API command to download the dataset (ensure your virtual environment is active):

kaggle datasets download -d hellbuoy/online-retail-customer-clustering -p data/raw --unzip


	6.	Run the project:
Execute the main script:

python main.py

The script will load, process, cluster, evaluate, and visualize the data. Results will be saved in the results/ folder.

Additional Information
	•	Clustering Model:
The project uses the KMeans algorithm for clustering and evaluates the model using the Silhouette Score. The ClusteringModel class has been updated to allow the user to specify the number of clusters during initialization.
	•	Cluster Profiling:
After clustering, cluster profiles (average values for each cluster) are generated and printed out. This helps in understanding the characteristics of each cluster.
	•	Visualization:
The Plotter class visualizes the clustering results using PCA for dimensionality reduction. The clusters are visualized in a 2D space to understand their distribution.
	•	Testing:
Run unit tests by executing the files in the tests/ directory to ensure all modules work as expected.
	•	Documentation:
For more details on each module, please refer to the comments in the source files. Additional documentation links for libraries like Pandas and scikit-learn can be added as needed.

License

This project is licensed under the MIT License - see the LICENSE file for details.