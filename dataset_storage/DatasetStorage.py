import logging
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class DatasetStorage:
    def __init__(self):
        """Initialize the connection to MongoDB using environment variables for connection string and database name."""
        # Load environment variables
        connection_string = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("DB_NAME", "anomaly_detection")
        self.base_folder_path = os.getenv("BASE_FOLDER_PATH", "dataset")

        # Connect to MongoDB
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[db_name]
            self.collection = self.db["datasets"]
            logging.info(f"Connected to MongoDB database '{db_name}' in collection 'datasets'.")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise

    def clear_collection(self):
        """Delete all entries in the 'datasets' collection."""
        try:
            result = self.collection.delete_many({})
            logging.info(f"Deleted {result.deleted_count} documents from 'datasets' collection.")
        except Exception as e:
            logging.error(f"Error deleting datasets from collection: {e}")

    def save_dataset(self, name, datasets):
        """Save a dataset to the MongoDB collection."""
        if not datasets or not isinstance(datasets, list) or len(datasets) == 0:
            logging.error("The datasets list must be a non-empty array.")
            return

        main_dataset_id = datasets[0].get("name", "undefined")

        document = {
            "name": name,
            "main_dataset_id": main_dataset_id,
            "datasets": datasets
        }

        try:
            result = self.collection.insert_one(document)
            logging.info(f"Dataset saved to MongoDB with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logging.error(f"Error saving dataset to MongoDB: {e}")
            return None

    def initialize_from_folder(self):
        """Initialize the dataset from the folder structure."""
        logging.info(f"Initializing dataset from base folder: {self.base_folder_path}")

        if not os.path.isdir(self.base_folder_path):
            logging.error(f"Base folder '{self.base_folder_path}' does not exist.")
            return

        for subdir in os.listdir(self.base_folder_path):
            subdir_path = os.path.join(self.base_folder_path, subdir)

            if os.path.isdir(subdir_path):
                datasets = []
                for root, dirs, files in os.walk(subdir_path):
                    directory_urls = [os.path.join(root, f) for f in files]

                    if directory_urls:
                        dataset_name = os.path.basename(root)
                        datasets.append({
                            "name": dataset_name,
                            "directory_urls": directory_urls
                        })

                if datasets:
                    self.save_dataset(subdir, datasets)
                else:
                    logging.warning(f"No datasets were found in the folder: {subdir_path}")

    def any_datasets_exist(self):
        """Check if any dataset exists in the MongoDB collection."""
        dataset = self.collection.find_one()
        if dataset:
            logging.info("At least one dataset exists in MongoDB.")
            return True
        else:
            logging.info("No datasets found in MongoDB.")
            return False
