import multiprocessing
import logging
from dataset_storage import DatasetStorage
from icp.icp import run_icp_analysis_parallel
from isolation_forest.isolation_forest import run_iso_forest_analysis_parallel
from dbscan.dbscan import run_dbscan_analysis_parallel  # Add this import
from knn.knn import run_knn_analysis_parallel
from library.dataset import load_datasets_from_mongo, load_datasets_ranked_from_mongo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_dataset():
    """Check if datasets exist, if not initialize from folder and run scripts."""
    # Initialize DatasetStorage instance
    dataset_storage = DatasetStorage()

    # Step 1: Check if any dataset exists in MongoDB
    if dataset_storage.any_datasets_exist():
        print("Datasets exist in MongoDB. Running the pipeline...")
    else:
        print("No datasets found. Initializing and running the pipeline...")

        # Initialize and save dataset (populate the database)
        dataset_storage.initialize_from_folder()

        print("Datasets initialized and saved.")


def run_icp():
    """Load datasets and run ICP analysis in parallel."""
    # Step 2: Load datasets from MongoDB
    datasets = load_datasets_from_mongo()

    if datasets:
        print(f"Running ICP analysis on {len(datasets)} datasets in parallel...")

        run_icp_analysis_parallel(datasets)


        print("ICP Pipeline completed.")
    else:
        print("No datasets found for ICP analysis.")


def run_iso_forest():
    """Load datasets and run Isolation Forest analysis in parallel."""
    # Step 2: Load datasets from MongoDB
    datasets = load_datasets_ranked_from_mongo()

    if datasets:
        print(f"Running Isolation Forest analysis on {len(datasets)} datasets in parallel...")

        # Initialize Isolation Forest detector
        from sklearn.ensemble import IsolationForest
        iso_forest_detector = IsolationForest(n_estimators=200, contamination=0.05)

        # Run Isolation Forest analysis for all datasets in parallel
        run_iso_forest_analysis_parallel(datasets, iso_forest_detector)

        print("Isolation Forest Pipeline completed.")
    else:
        print("No datasets found for Isolation Forest analysis.")


def run_dbscan():
    """Load ranked datasets and run DBSCAN analysis in parallel."""
    # Step 3: Load ranked datasets from MongoDB
    datasets = load_datasets_ranked_from_mongo()

    if datasets:
        print(f"Running DBSCAN analysis on {len(datasets)} ranked datasets in parallel...")

        # Run DBSCAN analysis for all ranked datasets in parallel
        run_dbscan_analysis_parallel(datasets, num_workers=8, voxel_size=0.01)

        print("DBSCAN Pipeline completed.")
    else:
        print("No ranked datasets found for DBSCAN analysis.")


def run_knn():
    datasets = load_datasets_ranked_from_mongo()

    if datasets:
        print(f"Running kNN analysis on {len(datasets)} ranked datasets in parallel...")

        run_knn_analysis_parallel(datasets, loc='01_Location', threshold_map=None, num_workers=8)

        print("KNN-Pipeline finished.")
    else:
        print("No ranked datasets found for DBSCAN analysis")

def main():
    """Main function to initialize dataset and run ICP, Isolation Forest, and DBSCAN analysis."""
    # Initialize datasets (checks if they exist or creates them)

    multiprocessing.set_start_method('spawn')
    initialize_dataset()  # Check and initialize datasets if necessary

    # Run ICP analysis on the datasets
    run_icp()  # Run ICP analysis on the datasets

    # Run Isolation Forest analysis on the datasets
    run_iso_forest()  # Run Isolation Forest analysis on the datasets

    # Run KNN analysis on the datasets
    run_knn()  # Run KNN analysis on the datasets

    # Run DBSCAN analysis on the ranked datasets
    run_dbscan()  # Run DBSCAN analysis on the ranked datasets


if __name__ == "__main__":
    main()
