import pandas as pd
from pymongo import MongoClient

from results.RankedResult import RankedResult


def analyse(
        mongo_uri="mongodb://localhost:27017/",
        db_name="anomaly_detection",
        collection_name="ranked_datasets",
        out_csv="ranked_datasets_export.csv"
):
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Load data
    ranked_results = []
    for doc in collection.find():
        try:
            ranked = RankedResult(
                dataset_name1=doc.get('dataset_name1', ''),
                sub_dataset_name1=doc.get('sub_dataset_name1', ''),
                dataset_name2=doc.get('dataset_name2', ''),
                sub_dataset_name2=doc.get('sub_dataset_name2', ''),
                scan1_path=doc.get('scan1_path', ''),
                scan2_path=doc.get('scan2_path', ''),
                residual_error=doc.get('residual_error', 0),
                total_rank=doc.get('total_rank', 0),
                icp_rank=doc.get('icp_rank', 0),
                dbscan_rank=doc.get('dbscan_rank', 0),
                knn_rank=doc.get('knn_rank', 0),
                iso_rank=doc.get('iso_rank', 0)
            )
            ranked_results.append(ranked)
        except KeyError as e:
            print(f"Skipping document: {e}")

    # Configuration
    TARGET_PATTERN = "01_Location"
    ALGORITHMS = [
        ('total_rank', 'TOTAL'),
        ('icp_rank', 'ICP'),
        ('dbscan_rank', 'DBSCAN'),
        ('knn_rank', 'KNN'),
        ('iso_rank', 'ISO')
    ]

    # Helper functions
    def get_rank(algorithm):
        return find_first_rank(
            sorted(ranked_results, key=lambda x: getattr(x, algorithm)),
            TARGET_PATTERN
        )

    def find_first_rank(sorted_list, pattern):
        for idx, item in enumerate(sorted_list, 1):
            if pattern in item.scan1_path and pattern in item.scan2_path:
                return idx
        return 0

    # Calculate ranks for 01_Location
    total_anomalies = len(ranked_results)
    location_pairs = sum(1 for r in ranked_results
                         if TARGET_PATTERN in r.scan1_path
                         and TARGET_PATTERN in r.scan2_path)
    non_location_pairs = total_anomalies - location_pairs

    analysis = {'total_anomalies': total_anomalies}

    location_ranks = {}
    for algo_key, algo_name in ALGORITHMS:
        rank = get_rank(algo_key)
        location_ranks[algo_key] = rank
        better_count = rank - 1 if rank > 0 else 0
        percentage = (better_count / non_location_pairs * 100) if non_location_pairs > 0 else 0

        analysis.update({
            f"{algo_key}": rank,
            f"{algo_key}_better": better_count,
            f"{algo_key}_pct": percentage
        })

    # Count how many comparisons are better in any algorithm
    any_better_than_location = 0
    for r in ranked_results:
        if TARGET_PATTERN in r.scan1_path and TARGET_PATTERN in r.scan2_path:
            continue  # skip the target pair itself

        for algo_key, _ in ALGORITHMS:
            current_rank = getattr(r, algo_key)
            if 0 < current_rank < location_ranks[algo_key]:
                any_better_than_location += 1
                break  # count only once per comparison

    analysis["better_than_01_location_in_any_algo"] = any_better_than_location

    # Print results
    print("\n=== Detailed Analysis ===")
    print(f"Total comparisons: {total_anomalies}")
    print(f"01_Location pairs: {location_pairs}")
    print(f"Other comparisons: {non_location_pairs}")
    print(f"Comparisons better than 01_Location in at least one algorithm: {any_better_than_location}\n")

    for algo_key, algo_name in ALGORITHMS:
        print(f"--- {algo_name} Results ---")
        print(f"Rank position: {analysis[algo_key]}")
        print(f"Anomalies ranked higher: {analysis[f'{algo_key}_better']}")
        print(f"Percentage of non-targets ranked higher: {analysis[f'{algo_key}_pct']:.2f}%\n")

    # Export to CSV
    pd.DataFrame([analysis]).to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    # Prozentualer Anteil der Vergleiche, die mindestens einmal besser waren
    if non_location_pairs > 0:
        pct_better_any = any_better_than_location / non_location_pairs * 100
    else:
        pct_better_any = 0.0

    analysis["better_than_01_location_in_any_algo_pct"] = pct_better_any

    print(f"Percentage of comparisons better than 01_Location in at least one algorithm: {pct_better_any:.2f}%\n")

if __name__ == "__main__":
    analyse()
