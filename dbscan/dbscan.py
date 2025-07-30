import os
import logging
import numpy as np
from pymongo import MongoClient
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import hdbscan
import hashlib
import json
import gc
from bson import ObjectId

from library.las import read_las
from library.time import format_time
from library.pointcloud import downsample_point_cloud

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DBSCANResult:
    """Stores DBSCAN comparison results between two scans."""

    def __init__(self, dataset_name1, sub_dataset_name1, dataset_name2, sub_dataset_name2,
                 scan1_path, scan2_path, metrics, rank=1, combined_score=0):
        self.dataset_name1 = dataset_name1
        self.sub_dataset_name1 = sub_dataset_name1
        self.dataset_name2 = dataset_name2
        self.sub_dataset_name2 = sub_dataset_name2
        self.scan1_path = scan1_path
        self.scan2_path = scan2_path
        self.metrics = metrics
        self.rank = rank
        self.combined_score = combined_score

    def __str__(self):
        s1 = os.path.basename(self.scan1_path)
        s2 = os.path.basename(self.scan2_path)
        return f"Dataset: {self.dataset_name1} | Sub-dataset: {self.sub_dataset_name1} | Scan 1: {s1}, Scan 2: {s2}"

    def to_dict(self):
        """Convert the DBSCANResult object to a dictionary for saving to MongoDB"""
        result = {
            "dataset_name1": self.dataset_name1,
            "sub_dataset_name1": self.sub_dataset_name1,
            "dataset_name2": self.dataset_name2,
            "sub_dataset_name2": self.sub_dataset_name2,
            "scan1_path": self.scan1_path,
            "scan2_path": self.scan2_path,
            "scan1_name": os.path.basename(self.scan1_path),
            "scan2_name": os.path.basename(self.scan2_path),
            "rank": self.rank,
            "combined_score": self.combined_score,
            "timestamp": datetime.now()
        }

        # Handle metrics conversion
        metric_keys = ['ari', 'avg_center_distance', 'silhouette1', 'silhouette2', 'noise_ratio_diff']
        if isinstance(self.metrics, list) and len(self.metrics) == len(metric_keys):
            self.metrics = dict(zip(metric_keys, self.metrics))

        result['metrics'] = self.metrics
        return result


def get_ranked_data_from_db(dataset_name=None):
    """Retrieve existing ranked data from MongoDB."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("DB_NAME", "anomaly_detection")

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection_ranked = db['ranked_datasets']

    query = {}
    if dataset_name:
        query = {
            "dataset_name1": dataset_name,
            "dataset_name2": dataset_name
        }

    ranked_docs = list(collection_ranked.find(query))
    logging.info(f"Retrieved {len(ranked_docs)} ranked documents from database")
    return ranked_docs


def update_dbscan_rank_by_id(doc_id, dbscan_rank):
    """Update the DBSCAN rank for a specific document by its _id."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("DB_NAME", "anomaly_detection")

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection_ranked = db['ranked_datasets']

    current_doc = collection_ranked.find_one({"_id": doc_id})
    if current_doc:
        current_total_rank = current_doc.get("total_rank", 0)
        new_total_rank = current_total_rank + dbscan_rank

        result = collection_ranked.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "dbscan_rank": dbscan_rank,
                    "total_rank": new_total_rank
                }
            }
        )

        if result.matched_count > 0:
            logging.info(f"Updated document {doc_id} with dbscan_rank {dbscan_rank} and total_rank {new_total_rank}")
            return True

    logging.error(f"Document with _id {doc_id} not found")
    return False


def generate_file_hash(file_path):
    """Generate a hash of the file content to use as cache identifier."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logging.error(f"Error generating hash for {file_path}: {e}")
        return None


def generate_cache_filename(scan_path, voxel_size):
    """Generate a unique cache file name based on file hash and voxel size."""
    hash_val = generate_file_hash(scan_path)
    if not hash_val:
        return None
    filename = os.path.basename(scan_path)
    return f"dbcache_{filename}_{hash_val}_{voxel_size}.json"


def generate_metrics_cache_filename(scan1_path, scan2_path, voxel_size):
    """Generate a consistent cache filename for the metrics."""
    scan_paths = sorted([scan1_path, scan2_path])
    hash1 = generate_file_hash(scan_paths[0])
    hash2 = generate_file_hash(scan_paths[1])
    if not hash1 or not hash2:
        return None
    return f"dbscanmetrics_{hash1}_{hash2}_{voxel_size}.json"


def load_cache(cache_file):
    """Load the cache from a file if it exists."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading cache from {cache_file}: {e}")
    return None


def save_cache(cache_file, cache_data):
    """Save the cache data to a file."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        logging.error(f"Error saving cache to {cache_file}: {e}")


def downsample_and_normalize(point_cloud, voxel_size):
    """Downsample and normalize point cloud."""
    downsampled = downsample_point_cloud(point_cloud, voxel_size=voxel_size, force_downsample=True)
    points = np.asarray(downsampled.points)
    points = MinMaxScaler().fit_transform(points)
    return points


def calculate_dbscan_metrics(scan1_path, scan2_path, voxel_size=0.01, cache_folder='cache/dbscancache',
                             metrics_cache_folder='cache/dbscanmetricscache'):
    """Calculate DBSCAN metrics for a pair of scans with caching."""
    try:
        #log both files
        logging.info("Metrics for:")
        logging.info(scan1_path)
        logging.info(scan2_path)
        start_time = time.time()
        os.makedirs(cache_folder, exist_ok=True)
        os.makedirs(metrics_cache_folder, exist_ok=True)

        cache_filename_1 = generate_cache_filename(scan1_path, voxel_size)
        cache_filename_2 = generate_cache_filename(scan2_path, voxel_size)
        metrics_cache_filename = generate_metrics_cache_filename(scan1_path, scan2_path, voxel_size)

        if not cache_filename_1 or not cache_filename_2 or not metrics_cache_filename:
            logging.error(f"Failed to generate cache filenames")
            return None

        cache_file_1 = os.path.join(cache_folder, cache_filename_1)
        cache_file_2 = os.path.join(cache_folder, cache_filename_2)
        metrics_cache_file = os.path.join(metrics_cache_folder, metrics_cache_filename)

        # Check metrics cache first
        metrics_cache = load_cache(metrics_cache_file)
        if metrics_cache:
            logging.info(
                f"Using cached DBSCAN metrics for {os.path.basename(scan1_path)} vs {os.path.basename(scan2_path)}")
            return metrics_cache

        # Check individual scan caches
        cache_1 = load_cache(cache_file_1)
        cache_2 = load_cache(cache_file_2)

        if cache_1 and cache_2:
            labels1 = cache_1.get('labels')
            labels2 = cache_2.get('labels')
            points1 = cache_1.get('points')
            points2 = cache_2.get('points')

            if labels1 and labels2 and points1 and points2:
                metrics = find_dbscan_thresholds(
                    np.array(labels1), np.array(points1),
                    np.array(labels2), np.array(points2)
                )
                save_cache(metrics_cache_file, metrics)
                logging.info(
                    f"Using cached DBSCAN metrics for {os.path.basename(scan1_path)} vs {os.path.basename(scan2_path)}")
                duration = time.time() - start_time  # ⏱ Dauer in Sekunden
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                logging.info(
                    f"DBSCAN Metrics between {scan1_path} and {scan2_path} took {minutes} min {seconds} sec")
                return metrics

        # Compute from scratch
        clusterer = hdbscan.HDBSCAN(min_samples=5, metric='euclidean')

        # Process first scan
        scan1 = read_las(scan1_path)
        points1 = downsample_and_normalize(scan1, voxel_size)
        clusterer.fit(points1)
        labels1 = clusterer.labels_

        # Process second scan
        scan2 = read_las(scan2_path)
        points2 = downsample_and_normalize(scan2, voxel_size)
        clusterer.fit(points2)
        labels2 = clusterer.labels_

        # Save individual caches
        save_cache(cache_file_1, {
            'labels': labels1.tolist(),
            'points': points1.tolist()
        })
        save_cache(cache_file_2, {
            'labels': labels2.tolist(),
            'points': points2.tolist()
        })

        # Calculate and save metrics
        metrics = find_dbscan_thresholds(labels1, points1, labels2, points2)
        save_cache(metrics_cache_file, metrics)
        duration = time.time() - start_time  # ⏱ Dauer in Sekunden
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        logging.info(
            f"DBSCAN Metrics between {scan1_path} and {scan2_path} took {minutes} min {seconds} sec")

        return metrics

    except Exception as e:
        logging.error(f"Error in DBSCAN computation for {scan1_path} and {scan2_path}: {e}")
        return None


def normalize_metric(value, min_value, max_value):
    """Normalize a value to be between 0 and 1."""
    return (value - min_value) / (max_value - min_value) if max_value != min_value else 0


def calculate_combined_score(metrics_list):
    """Calculate combined scores for ranking."""
    metric_names = ['ari', 'avg_center_distance', 'silhouette1', 'silhouette2', 'noise_ratio_diff']

    min_values = {metric: float('inf') for metric in metric_names}
    max_values = {metric: float('-inf') for metric in metric_names}

    # Find min and max values
    for metrics in metrics_list:
        if metrics:
            for metric in metric_names:
                if isinstance(metrics, list) and len(metrics) >= 5:
                    metric_dict = dict(zip(metric_names, metrics))
                    metric_value = metric_dict.get(metric, 0)
                else:
                    metric_value = metrics.get(metric, 0)

                min_values[metric] = min(min_values[metric], metric_value)
                max_values[metric] = max(max_values[metric], metric_value)

    # Calculate combined scores
    combined_scores = []
    for metrics in metrics_list:
        if metrics:
            normalized_score = 0
            for metric in metric_names:
                if isinstance(metrics, list) and len(metrics) >= 5:
                    metric_dict = dict(zip(metric_names, metrics))
                    metric_value = metric_dict.get(metric, 0)
                else:
                    metric_value = metrics.get(metric, 0)

                normalized_value = normalize_metric(metric_value, min_values[metric], max_values[metric])
                normalized_score += normalized_value

            combined_scores.append(normalized_score)
        else:
            combined_scores.append(0)

    return combined_scores


def write_dbscan_results_to_mongodb(dbscan_results, dataset_name):
    """Write DBSCAN analysis results to MongoDB collection."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("DB_NAME", "anomaly_detection")

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection_dbscan = db['dbscan_results']

    # Clear previous DBSCAN results for this dataset
    collection_dbscan.delete_many({"dataset_name1": dataset_name, "dataset_name2": dataset_name})

    # Insert new results
    if dbscan_results:
        docs_to_insert = [result.to_dict() for result in dbscan_results]
        collection_dbscan.insert_many(docs_to_insert)
        logging.info(f"Inserted {len(docs_to_insert)} DBSCAN results into MongoDB")
        return len(docs_to_insert)

    return 0


def process_ranked_document(ranked_doc, voxel_size=0.01, cache_folder='cache/dbscancache',
                            metrics_cache_folder='cache/dbscanmetricscache'):
    """Process a single ranked document using DBSCAN with caching."""
    try:
        start_time = time.time()
        scan1_path = ranked_doc['scan1_path']
        scan2_path = ranked_doc['scan2_path']
        doc_id = ranked_doc['_id']

        # Calculate DBSCAN metrics with caching
        metrics = calculate_dbscan_metrics(scan1_path, scan2_path, voxel_size, cache_folder, metrics_cache_folder)
        duration = time.time() - start_time  # ⏱ Dauer in Sekunden
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        logging.info(
            f"DBSCAN between {scan1_path} and {scan2_path} took {minutes} min {seconds} sec")

        if metrics is None:
            logging.error(f"Failed to calculate DBSCAN metrics for document {doc_id}")
            return None

        # Create DBSCANResult object
        dbscan_result = DBSCANResult(
            ranked_doc['dataset_name1'], ranked_doc['sub_dataset_name1'],
            ranked_doc['dataset_name2'], ranked_doc['sub_dataset_name2'],
            scan1_path, scan2_path, metrics
        )

        return {
            'doc_id': doc_id,
            'metrics': metrics,
            'dbscan_result': dbscan_result,
            'scan1_path': scan1_path,
            'scan2_path': scan2_path
        }

    except Exception as e:
        logging.error(f"Error processing document {ranked_doc['_id']}: {e}")
        return None


def run_dbscan_analysis_parallel(ranked_docs, num_workers=None, voxel_size=0.01, cache_folder='cache/dbscancache',
                                 metrics_cache_folder='cache/dbscanmetricscache'):
    """Run DBSCAN analysis on existing ranked data and update dbscan_rank field + insert into dbscan_results."""
    start_time = time.time()

    if not num_workers:
        num_workers = min(14, multiprocessing.cpu_count())

    os.makedirs(cache_folder, exist_ok=True)
    os.makedirs(metrics_cache_folder, exist_ok=True)

    logging.info(f"Using HDBSCAN with {num_workers} workers and voxel_size {voxel_size}")

    if not ranked_docs:
        logging.error("No ranked documents found in database")
        return []

    # Process documents in parallel
    processed_results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_doc = {
            executor.submit(process_ranked_document, doc, voxel_size, cache_folder, metrics_cache_folder): doc
            for doc in ranked_docs
        }

        for future in as_completed(future_to_doc):
            try:
                result = future.result(timeout=300)
            except TimeoutError:
                doc = future_to_doc[future]
                logging.warning(f"Timeout on document {doc['_id']}")
                result = {
                    'doc_id': doc['_id'],
                    'metrics': {},
                    'combined_score': -1, #Ensure Scans which could not get a combined scan are ranked very low
                    'dbscan_result': None
                }
            if result:
                processed_results.append(result)

            completed += 1
            elapsed = time.time() - start_time
            elapsed_time = format_time(elapsed)
            progress = completed / len(ranked_docs) * 100
            logging.info(f"Progress: {completed}/{len(ranked_docs)} ({progress:.1f}%) - Time elapsed: {elapsed_time}")

    # Calculate combined scores and assign ranks
    metrics_list = [result['metrics'] for result in processed_results]
    combined_scores = calculate_combined_score(metrics_list)

    for i, result in enumerate(processed_results):
        result['combined_score'] = combined_scores[i]
        if result.get('dbscan_result'):
            result['dbscan_result'].combined_score = combined_scores[i]

    # Sort by combined score (descending) and assign ranks
    processed_results.sort(key=lambda x: x['combined_score'], reverse=True)

    dbscan_results = []
    for rank, result in enumerate(processed_results, 1):
        if result.get('dbscan_result'):
            result['dbscan_result'].rank = rank
            dbscan_results.append(result['dbscan_result'])

    # Get dataset name for saving results
    dataset_name = None
    if dbscan_results:
        dataset_name = dbscan_results[0].dataset_name1

    # 1. INSERT into dbscan_results collection
    if dataset_name:
        write_dbscan_results_to_mongodb(dbscan_results, dataset_name)

    # 2. UPDATE ranked_datasets collection
    updated_count = 0
    for rank, result in enumerate(processed_results, 1):
        success = update_dbscan_rank_by_id(result['doc_id'], rank)
        if success:
            updated_count += 1
            logging.info(
                f"Rank {rank}: {os.path.basename(result['scan1_path'])} & {os.path.basename(result['scan2_path'])} - Combined Score: {result['combined_score']:.6f}")

    total_time = time.time() - start_time
    logging.info(f"\nCompleted processing {len(ranked_docs)} documents in {format_time(total_time)}")
    logging.info(f"Successfully updated {updated_count} documents with new dbscan_rank values")

    return processed_results


def find_dbscan_thresholds(dbscan_labels1, points1, dbscan_labels2, points2):
    """
    Retrieve DBSCAN comparison metrics without checking the predefined criteria.

    Args:
        dbscan_labels1 (list): Cluster labels for the first dataset.
        points1 (list): Points associated with the first dataset.
        dbscan_labels2 (list): Cluster labels for the second dataset.
        points2 (list): Points associated with the second dataset.

    Returns:
        tuple: A tuple containing ARI, avg_center_distance, silhouette1, silhouette2, and noise_ratio_diff.
    """
    metrics = calculate_dbscan_metrics(dbscan_labels1, points1, dbscan_labels2, points2)

    if not metrics.get('is_valid', False):
        return None  # Return None if there's an invalid comparison

    return (
        metrics['ari'],
        metrics['avg_center_distance'],
        metrics['silhouette1'],
        metrics['silhouette2'],
        metrics['noise_ratio_diff']
    )
