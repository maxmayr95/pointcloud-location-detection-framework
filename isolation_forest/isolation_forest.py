import os
import logging
import numpy as np
from pymongo import MongoClient
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from library.las import read_las
from library.time import format_time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_ranked_data_from_db(dataset_name=None):
    """Retrieve existing ranked data from MongoDB."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("DB_NAME", "anomaly_detection")

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection_ranked = db['ranked_datasets']

    # Build query filter
    query = {}
    if dataset_name:
        query = {
            "dataset_name1": dataset_name,
            "dataset_name2": dataset_name
        }

    # Retrieve all ranked documents
    ranked_docs = list(collection_ranked.find(query))
    logging.info(f"Retrieved {len(ranked_docs)} ranked documents from database")

    return ranked_docs


def update_iso_rank_by_id(doc_id, iso_rank):
    """Update the Isolation Forest rank for a specific document by its _id."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("DB_NAME", "anomaly_detection")

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection_ranked = db['ranked_datasets']

    # First, get the current document to read total_rank
    current_doc = collection_ranked.find_one({"_id": doc_id})

    if current_doc:
        current_total_rank = current_doc.get("total_rank", 0)
        new_total_rank = current_total_rank + iso_rank

        # Update both fields with calculated values
        result = collection_ranked.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "iso_rank": iso_rank,
                    "total_rank": new_total_rank
                }
            }
        )

        if result.matched_count > 0:
            logging.info(f"Updated document {doc_id} with iso_rank {iso_rank} and total_rank {new_total_rank}")
            return True

    logging.error(f"Document with _id {doc_id} not found")
    return False


def process_ranked_document(ranked_doc, iso_forest_detector):
    """Process a single ranked document using Isolation Forest."""
    try:
        scan1_path = ranked_doc['scan1_path']
        scan2_path = ranked_doc['scan2_path']
        doc_id = ranked_doc['_id']

        # Read LAS files
        scan1 = read_las(scan1_path)
        scan2 = read_las(scan2_path)
        points1 = np.asarray(scan1.points)
        points2 = np.asarray(scan2.points)

        # Fit and score both scans using Isolation Forest
        iso_forest_detector.fit(points1)
        iso_scores1 = iso_forest_detector.decision_function(points1)
        iso_forest_detector.fit(points2)
        iso_scores2 = iso_forest_detector.decision_function(points2)

        # Use the minimum anomaly score as the pair's score
        min_iso_score = min(np.min(iso_scores1), np.min(iso_scores2))

        # Return document ID, ISO score, and paths for logging
        return {
            'doc_id': doc_id,
            'iso_score': min_iso_score,
            'scan1_path': scan1_path,
            'scan2_path': scan2_path
        }

    except Exception as e:
        logging.error(f"Error processing document {ranked_doc['_id']}: {e}")
        return None


def run_iso_forest_analysis_parallel(ranked_docs, iso_forest_detector=None, num_workers=None):
    """Run Isolation Forest analysis on existing ranked data and update iso_rank field."""
    start_time = time.time()

    # Set default number of workers
    if not num_workers:
        num_workers = min(14, multiprocessing.cpu_count())

    # Set default Isolation Forest detector

    logging.info(f"Using Isolation Forest with {iso_forest_detector.n_estimators} estimators")


    if not ranked_docs:
        logging.error("No ranked documents found in database")
        return []

    # Process documents in parallel
    processed_results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_doc = {
            executor.submit(process_ranked_document, doc, iso_forest_detector): doc
            for doc in ranked_docs
        }

        # Collect results as they complete
        for future in as_completed(future_to_doc):
            result = future.result()
            if result:
                processed_results.append(result)

            completed += 1
            elapsed = time.time() - start_time
            elapsed_time = format_time(elapsed)
            progress = completed / len(ranked_docs) * 100
            logging.info(f"Progress: {completed}/{len(ranked_docs)} ({progress:.1f}%) - Time elapsed: {elapsed_time}")

    # Sort results by ISO score (descending) to assign ranks
    processed_results.sort(key=lambda x: x['iso_score'], reverse=True)

    # Update database with new ranks
    updated_count = 0
    for rank, result in enumerate(processed_results, 1):
        success = update_iso_rank_by_id(result['doc_id'], rank)
        if success:
            updated_count += 1
            logging.info(
                f"Rank {rank}: {os.path.basename(result['scan1_path'])} & {os.path.basename(result['scan2_path'])} - ISO Score: {result['iso_score']:.6f}")
        else:
            logging.error(f"Failed to update rank for document {result['doc_id']}")
    total_time = time.time() - start_time
    logging.info(f"\nCompleted processing {len(ranked_docs)} documents in {format_time(total_time)}")
    logging.info(f"Successfully updated {updated_count} documents with new iso_rank values")

    return processed_results




