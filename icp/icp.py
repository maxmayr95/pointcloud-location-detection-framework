import multiprocessing
import os
import logging
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
import time


import numpy as np
from pymongo import MongoClient

from library.dataset import get_las_files_from_dataset
from library.las import read_las
from library.time import format_time
from results.ICPResult import ICPResult
from results.RankedResult import RankedResult
import open3d as o3d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def apply_icp(source, target):
    """Apply ICP to align two point clouds and calculate the residual error."""
    log_scans(source, target)  # Log the scans being compared
    threshold = 0.01
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg_p2p.transformation, reg_p2p.inlier_rmse




def log_scans(scan1, scan2):
    """Log the details of the scans being compared."""
    logging.info(f"Scan 1: {scan1}")
    logging.info(f"Scan 2: {scan2}")
    logging.info(f"Number of points in Scan 1: {len(scan1.points)}")
    logging.info(f"Number of points in Scan 2: {len(scan2.points)}")


def compare_scan_pair(pair):
    """Compare a pair of scans using ICP and return the result."""
    (dataset_name1, sub_dataset_name1, scan1_path), (dataset_name2, sub_dataset_name2, scan2_path) = pair
    try:
        scan1 = read_las(scan1_path)
        scan2 = read_las(scan2_path)
        if np.array_equal(scan1.points, scan2.points):
            logging.error("Identical point clouds detected!")

        # Apply ICP and get residual error
        _, residual_error = apply_icp(scan1, scan2)

        if residual_error == 0:
            logging.warning(f"Residual error is zero for {os.path.basename(scan1_path)} and {os.path.basename(scan2_path)}. This may indicate identical scans or an issue with the ICP algorithm.")
            residual_error = -1 # Set a high residual error for identical scans to avoid misleading results
        # Create an ICPResult instance and prepare for saving in MongoDB
        result = ICPResult(dataset_name1, sub_dataset_name1, dataset_name2, sub_dataset_name2,
                           scan1_path, scan2_path, residual_error)

        # Create a RankedResult instance and prepare for saving in MongoDB
        ranked_result = RankedResult(dataset_name1, sub_dataset_name1, dataset_name2, sub_dataset_name2,
                                     scan1_path, scan2_path, residual_error, total_rank=0, icp_rank=0, dbscan_rank=0, knn_rank=0)
        logging.info(f"Successfully compared {scan1_path} and {scan2_path} with residual error {residual_error}")
        return result, ranked_result
    except Exception as e:
        logging.error(f"Error comparing {os.path.basename(scan1_path)} and {os.path.basename(scan2_path)}: {e}")
        return None, None


def write_results_to_mongodb(results, ranked_results, dataset_name):
    """Write ICP analysis results to MongoDB collection."""
    # Get MongoDB connection details from the environment
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("DB_NAME", "anomaly_detection")

    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    icp_collection = db['icp_results']
    ranked_collection = db['ranked_datasets']

    # Combine results and ranked_results for synchronized sorting
    combined = list(zip(results, ranked_results))
    combined.sort(key=lambda x: x[0].residual_error, reverse=True)  # Sort desc (smaller error is better)

    # Clear old lists
    results = []
    ranked_results = []
    icp_result_docs = []
    ranked_result_docs = []

    # Assign ranks and build documents
    for rank, (result, ranked_result) in enumerate(combined, 1):
        result.total_rank = rank
        result.icp_rank = rank

        ranked_result.total_rank = rank
        ranked_result.icp_rank = rank
        ranked_result.residual_error = result.residual_error  # Ensure consistency

        results.append(result)
        ranked_results.append(ranked_result)

        icp_result_docs.append(result.to_dict())
        ranked_result_docs.append(ranked_result.to_dict())

    # Insert into MongoDB
    if icp_result_docs:
        icp_collection.insert_many(icp_result_docs)
        logging.info(f"Inserted {len(icp_result_docs)} results into MongoDB collection 'icp_results'")
    else:
        logging.warning("No ICP results to insert into MongoDB")

    if ranked_result_docs:
        ranked_collection.insert_many(ranked_result_docs)
        logging.info(f"Inserted {len(ranked_result_docs)} results into MongoDB collection 'ranked_datasets'")
    else:
        logging.warning("No ranked results to insert into MongoDB")

    return len(icp_result_docs)



def run_icp_analysis_parallel(datasets, num_workers=None, max_scans_per_location=None):
    """Run ICP analysis on scan pairs from multiple datasets in parallel and save results to MongoDB."""
    start_time = time.time()

    all_scan_pairs = []
    for dataset in datasets:
        las_files = get_las_files_from_dataset(dataset, max_scans_per_location)
        if not las_files:
            logging.error(f"No LAS files found for dataset {dataset['name']}")
            continue

        scan_pairs = [(file1, las_files[j]) for i, file1 in enumerate(las_files) for j in range(i + 1, len(las_files))]
        all_scan_pairs.append((dataset, scan_pairs))

    results = []
    ranked_results = []
    completed = 0

    if not num_workers:
        num_workers = min(14, multiprocessing.cpu_count())

    with ProcessPoolExecutor(max_workers=num_workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        future_to_pair = {}

        for dataset, scan_pairs in all_scan_pairs:
            for pair in scan_pairs:
                future = executor.submit(compare_scan_pair, pair)
                future_to_pair[future] = dataset

        for future in as_completed(future_to_pair):
            result, ranked_result = future.result()
            if result and ranked_result:
                results.append(result)
                ranked_results.append(ranked_result)
            completed += 1

            elapsed = time.time() - start_time
            elapsed_time = format_time(elapsed)
            progress = completed / len(future_to_pair) * 100
            logging.info(f"Progress: {completed}/{len(future_to_pair)} ({progress:.1f}%) - Time elapsed: {elapsed_time}")

    #If ranked result or result has a residual error of "0" make the rank last



    total_entries = write_results_to_mongodb(results, ranked_results, dataset_name="icp_dataset")

    logging.info(f"\nResults saved to MongoDB collection 'icp_results'. Total entries: {total_entries}")
    return results
