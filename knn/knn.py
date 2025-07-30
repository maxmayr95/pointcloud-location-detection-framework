import os
import logging
import json
import hashlib
import numpy as np
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from library.las import read_las
from library.time import format_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("knn_analysis.log")
    ]
)

THRESHOLD = 5


def get_ranked_data_from_db(dataset_name=None):
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
    logging.info(f"Retrieved {len(ranked_docs)} ranked documents for KNN analysis")
    return ranked_docs


def update_knn_rank(doc_id, knn_rank):
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("DB_NAME", "anomaly_detection")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection_ranked = db['ranked_datasets']
    current_doc = collection_ranked.find_one({"_id": doc_id})
    if current_doc:
        current_total_rank = current_doc.get("total_rank", 0)
        new_total_rank = current_total_rank + knn_rank
        result = collection_ranked.update_one(
            {"_id": doc_id},
            {"$set": {
                "knn_rank": knn_rank,
                "total_rank": new_total_rank
            }}
        )
        if result.modified_count > 0:
            logging.info(f"Updated KNN rank {knn_rank} for {doc_id}")
            return True
    return False


def calculate_knn_value(points, k=5, distance_threshold=THRESHOLD):
    if len(points) <= k:
        k = max(1, len(points) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, _ = nn.kneighbors(points)
    distances = distances[:, 1:]
    with np.errstate(invalid='ignore'):
        filtered = np.where(distances <= distance_threshold, distances, np.nan)
        avg_distances = np.nanmean(filtered, axis=1)
    avg_distances[np.isnan(avg_distances)] = distance_threshold + 1e-6
    return np.max(avg_distances)


def compute_location_avg(ranked_docs, location_name="01_Location"):
    logging.info(f"Scanning for scan pairs where BOTH scans are from '{location_name}'...")

    scan_paths_with_thresholds = []
    for doc in ranked_docs:
        if location_name in doc.get('sub_dataset_name1', '') and location_name in doc.get('sub_dataset_name2', ''):
            scan_paths_with_thresholds.append((doc['scan1_path'], doc.get('threshold1', THRESHOLD)))
            scan_paths_with_thresholds.append((doc['scan2_path'], doc.get('threshold2', THRESHOLD)))

    if not scan_paths_with_thresholds:
        raise ValueError(f"No scan pairs found where both scans are from location '{location_name}'")

    os.makedirs("cache", exist_ok=True)
    hash_source = "".join(sorted([path for path, _ in scan_paths_with_thresholds]))
    hash_key = hashlib.md5(hash_source.encode()).hexdigest()
    cache_path = f"cache/location1_avg_{hash_key}.json"

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            avg = json.load(f)['avg']
            logging.info(f"✅ Loaded cached location1_avg = {avg:.6f} from {cache_path}")
            return avg

    logging.info(f"Found {len(scan_paths_with_thresholds)} scans to process for average KNN calculation.")
    values = []
    start_time = time.time()

    for i, (scan_path, threshold) in enumerate(scan_paths_with_thresholds, 1):
        try:
            points = np.asarray(read_las(scan_path).points)
            knn_val = calculate_knn_value(points, distance_threshold=threshold)
            values.append(knn_val)
            elapsed = time.time() - start_time
            est_total = (elapsed / i) * len(scan_paths_with_thresholds)
            remaining = est_total - elapsed
            logging.info(f"[{i}/{len(scan_paths_with_thresholds)}] {os.path.basename(scan_path)}: KNN={knn_val:.6f} "
                         f"| {format_time(elapsed)} elapsed, {format_time(remaining)} remaining")
        except Exception as e:
            logging.warning(f"Skipping scan {scan_path}: {e}")

    if not values:
        raise ValueError(f"No valid KNN values for location '{location_name}'.")

    avg = np.mean(values)
    logging.info(f"✅ Computed final location1_avg = {avg:.6f} from {len(values)} scans.")
    with open(cache_path, 'w') as f:
        json.dump({"avg": avg}, f)
    return avg


def process_ranked_document(ranked_doc, location1_avg):
    try:
        scan1_path = ranked_doc['scan1_path']
        scan2_path = ranked_doc['scan2_path']
        doc_id = ranked_doc['_id']
        scan1 = read_las(scan1_path)
        scan2 = read_las(scan2_path)
        t1 = ranked_doc.get('threshold1', THRESHOLD)
        t2 = ranked_doc.get('threshold2', THRESHOLD)
        points1 = np.asarray(scan1.points)
        points2 = np.asarray(scan2.points)
        if points1.size == 0 or points2.size == 0:
            logging.warning(f"Empty scan: {scan1_path} or {scan2_path}")
            return None
        kv1 = calculate_knn_value(points1, distance_threshold=t1)
        kv2 = calculate_knn_value(points2, distance_threshold=t2)
        similarity_score = (kv1 + kv2) / 2
        location_avg_diff = abs(location1_avg - similarity_score)
        logging.info(f"[KNN] {os.path.basename(scan1_path)} vs {os.path.basename(scan2_path)} | "
                     f"kv1={kv1:.6f}, kv2={kv2:.6f} -> score={similarity_score:.6f}, diff={location_avg_diff:.6f}")
        return {
            'doc_id': doc_id,
            'knn_score': similarity_score,
            'location_avg_diff': location_avg_diff,
            'scan1_path': scan1_path,
            'scan2_path': scan2_path
        }
    except Exception as e:
        logging.error(f"Error in document {ranked_doc['_id']}: {e}")
        return None


def run_knn_analysis_parallel(ranked_docs,loc, threshold_map=None, num_workers=None):
    logging.info("Starting KNN analysis...")
    start_time = time.time()
    if not num_workers:
        num_workers = min(14, multiprocessing.cpu_count())
    if threshold_map:
        logging.info("Applying threshold map to documents...")
        for doc in ranked_docs:
            sub1 = doc.get('sub_dataset_name1', '')
            sub2 = doc.get('sub_dataset_name2', '')
            doc['threshold1'] = threshold_map.get(sub1, THRESHOLD)
            doc['threshold2'] = threshold_map.get(sub2, THRESHOLD)
    location1_avg = compute_location_avg(ranked_docs,location_name=loc)
    processed_results = []
    completed = 0
    compare_func = partial(process_ranked_document, location1_avg=location1_avg)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_doc = {executor.submit(compare_func, doc): doc for doc in ranked_docs}
        for future in as_completed(future_to_doc):
            result = future.result()
            if result:
                processed_results.append(result)
            completed += 1
            if completed % 5 == 0 or completed == len(ranked_docs):
                elapsed = time.time() - start_time
                logging.info(f"Progress: {completed}/{len(ranked_docs)} ({completed / len(ranked_docs) * 100:.1f}%) - {format_time(elapsed)}")
    processed_results.sort(key=lambda x: x['knn_score'], reverse=True)
    updated_count = 0
    for rank, result in enumerate(processed_results, 1):
        if update_knn_rank(result['doc_id'], rank):
            updated_count += 1
            logging.info(f"Updated rank {rank} for document {result['doc_id']}")
    total_time = time.time() - start_time
    logging.info(f"KNN analysis completed: {len(processed_results)} results in {format_time(total_time)}")
    logging.info(f"Updated ranks in {updated_count} documents")
    write_knn_results_to_mongo(processed_results, ranked_docs[0]['dataset_name1'])
    return processed_results


def write_knn_results_to_mongo(processed_results, dataset_name):
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("DB_NAME", "anomaly_detection")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection_knn = db['knn_results']
    logging.info(f"Cleaning old records for dataset '{dataset_name}' from knn_results...")
    collection_knn.delete_many({"dataset_name": dataset_name})
    docs = []
    for rank, result in enumerate(sorted(processed_results, key=lambda x: x['knn_score'], reverse=True), 1):
        docs.append({
            "rank": rank,
            "dataset_name": dataset_name,
            "scan1_path": result['scan1_path'],
            "scan2_path": result['scan2_path'],
            "knn_score": result['knn_score'],
            "location_avg_diff": result['location_avg_diff'],
            "timestamp": time.time()
        })
    if docs:
        collection_knn.insert_many(docs)
        logging.info(f"Inserted {len(docs)} results into knn_results.")
    else:
        logging.warning("No KNN results to insert into knn_results.")
