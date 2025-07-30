import logging
import os

from pymongo import MongoClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_datasets_from_mongo():
    """Load datasets from MongoDB for Isolation Forest analysis."""

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")  # Default to localhost if not set
    db_name = os.getenv("DB_NAME", "anomaly_detection") # Default to 'anomaly_detection' if not set
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db['datasets']

    datasets = list(collection.find())
    logging.info(f"Found {len(datasets)} datasets in the MongoDB collection.")
    return datasets

def load_datasets_ranked_from_mongo():
    """Load datasets from MongoDB for Isolation Forest analysis."""

    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")  # Default to localhost if not set
    db_name = os.getenv("DB_NAME", "anomaly_detection") # Default to 'anomaly_detection' if not set
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db['ranked_datasets']

    datasets = list(collection.find())
    logging.info(f"Found {len(datasets)} ranked_datasets in the MongoDB collection.")
    return datasets


def get_las_files_from_dataset(dataset, max_files_per_location=None):
    """Get LAS files from the dataset."""
    las_files = []
    for sub_dataset in dataset['datasets']:
        sub_dataset_name = sub_dataset.get('name', 'Unnamed')
        directory_urls = sub_dataset.get('directory_urls', [])
        # Filter for .las files
        las_urls = [url for url in directory_urls if url.lower().endswith('.las')]
        if max_files_per_location is not None:
            las_urls = las_urls[:max_files_per_location]
        for url in las_urls:
            las_files.append((dataset['name'], sub_dataset_name, url))
    return sorted(las_files)