import itertools
import logging
import pandas as pd
from dns.dnssecalgs import algorithms
from pymongo import MongoClient
from inspect import signature
from results.RankedResult import RankedResult

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def filter_dict_for_class(cls, data_dict):
    params = signature(cls.__init__).parameters
    valid_keys = set(params.keys()) - {'self'}
    return {k: v for k, v in data_dict.items() if k in valid_keys}


class AnalysisPipeline:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="anomaly_detection"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.TARGET_PATTERN = "01_Location"
        self.ALGORITHMS = [
            ('total_rank', 'TOTAL'),
            ('icp_rank', 'ICP'),
            ('dbscan_rank', 'DBSCAN'),
            ('knn_rank', 'KNN'),
            ('iso_rank', 'ISO')
        ]

    def load_ranked_data(self):
        collection = self.db['ranked_datasets']
        results = []
        for doc in collection.find({}, {'_id': False}):
            try:
                filtered_doc = filter_dict_for_class(RankedResult, doc)
                results.append(RankedResult(**filtered_doc))
            except Exception as e:
                logging.error(f"Fehler beim Laden des Dokuments: {e}")
        return results

    def analyze_results(self,algorithms, out_csv="ranked_datasets_export.csv"):
        ranked_results = self.load_ranked_data()
        if not ranked_results:
            logging.error("No data found!")
            return

        analysis = {
            'total_anomalies': len(ranked_results),
            'location_pairs': sum(1 for r in ranked_results
                                  if self.TARGET_PATTERN in r.scan1_path and self.TARGET_PATTERN in r.scan2_path)
        }
        analysis['non_location_pairs'] = analysis['total_anomalies'] - analysis['location_pairs']

        for algo_key in algorithms:
            sorted_results = sorted(ranked_results, key=lambda x: getattr(x, algo_key))
            try:
                rank = next(i + 1 for i, r in enumerate(sorted_results)
                            if self.TARGET_PATTERN in r.scan1_path and self.TARGET_PATTERN in r.scan2_path)
            except StopIteration:
                rank = 0
            analysis.update({
                f"{algo_key}": rank,
                f"{algo_key}_better": max(rank - 1, 0),
                f"{algo_key}_pct": (max(rank - 1, 0) / analysis['non_location_pairs'] * 100
                                    if analysis['non_location_pairs'] > 0 else 0)
            })

        analysis["better_than_01_location_in_any_algo"] = sum(
            1 for r in ranked_results
            if self.TARGET_PATTERN not in r.scan1_path and self.TARGET_PATTERN not in r.scan2_path and
            any(getattr(r, algo) > analysis[algo] for algo in algorithms)
        )

        self._save_and_print_results(analysis,algorithms, out_csv)
        return analysis

    def _save_and_print_results(self, analysis,algorithms, out_csv):
        df = pd.DataFrame([analysis])
        df.to_csv(out_csv, index=False)

        print("\n=== Detailed Analysis ===")
        print(f"Total comparisons: {analysis['total_anomalies']}")
        print(f"01_Location pairs: {analysis['location_pairs']}")
        print(f"Other comparisons: {analysis['non_location_pairs']}")
        print(f"Better in at least one algorithm: {analysis['better_than_01_location_in_any_algo']}")
        if analysis['non_location_pairs'] > 0:
            print(
                f"Percentage: {analysis['better_than_01_location_in_any_algo'] / analysis['non_location_pairs'] * 100:.2f}%\n")
        else:
            print("Percentage: No other comparisons available.\n")

        for algo_key in algorithms:
            algo_name = next((name for key, name in self.ALGORITHMS if key == algo_key), algo_key)
            print(f"--- {algo_name} Results ---")
            print(f"Rank position: {analysis[algo_key]} (higher = better)")
            print(f"Anomalies ranked worse: {analysis[f'{algo_key}_better']}")
            if analysis['non_location_pairs'] > 0:
                print(f"Percentage of non-target pairs ranked worse: {analysis[f'{algo_key}_pct']:.2f}%\n")
            else:
                print("Percentage of non-target pairs ranked worse: No other comparisons available.\n")

    def weighted_aggregation(self, algorithms, out_csv="weighted_rank_export.csv"):
        print(f"\n🧪 Starting weighted aggregation with algorithms: {algorithms}")
        ranked_results = self.load_ranked_data()
        if not ranked_results:
            logging.error("No data available.")
            return

        df = pd.DataFrame([r.to_dict() for r in ranked_results])
        print(f"📄 Loaded datasets: {len(df)}")

        # Automatically determine goal-based weighting
        weights = self.determine_goal_based_weights(algorithms)
        if weights is None:
            logging.error("Weight determination failed.")
            return

        logging.info(f"✅ Weights used for aggregation: {weights}")

        # Normalize the columns
        for col in weights:
            max_val = df[col].max() if df[col].max() != 0 else 1
            df[f'{col}_norm'] = df[col] / max_val
            print(f"🔧 Normalizing {col}: Max={max_val}, First values: {df[f'{col}_norm'].head(3).tolist()}")

        # Calculate weighted sum
        df['weighted_score'] = sum(df[f'{col}_norm'] * weight for col, weight in weights.items())
        print(f"📊 Calculated weighted scores: {df['weighted_score'].describe()}")

        # Filter out invalid scores
        df_clean = df[df['weighted_score'].notna() & df['weighted_score'].apply(lambda x: x != float('inf'))]
        df_clean['weighted_rank'] = df_clean['weighted_score'].rank(ascending=False).astype(int)
        print(f"✅ Filtered to valid scores: {len(df_clean)} datasets")

        # 📌 Create sorted dataset for analysis
        df_sorted = df_clean.sort_values('weighted_rank', ascending=True).reset_index(drop=True)
        df_sorted.to_csv(out_csv, index=False)
        print(f"💾 Results saved to: {out_csv}")

        # 🔍 Analyze final sorted dataset
        total = len(df_sorted)
        print(f"\n🔎 Total number of weighted results: {total}")

        mask_scan1 = df_sorted['scan1_path'].str.contains(self.TARGET_PATTERN)
        mask_scan2 = df_sorted['scan2_path'].str.contains(self.TARGET_PATTERN)

        location_both_sides = df_sorted[mask_scan1 & mask_scan2]
        location_only_one_side = df_sorted[mask_scan1 ^ mask_scan2]
        location_none = df_sorted[~mask_scan1 & ~mask_scan2]

        print(f"📍 Both sides '01_Location': {len(location_both_sides)}")
        print(f"📍 One side '01_Location': {len(location_only_one_side)}")
        print(f"📍 No '01_Location': {len(location_none)}")

        if total > 0:
            percentage_one_side = (len(location_only_one_side) / total) * 100
            print(f"➡️ Percentage one side: {percentage_one_side:.2f}%")

        if not location_both_sides.empty:
            print("\n🏅 Best pair with '01_Location' on both sides:")
            print(location_both_sides[['scan1_path', 'scan2_path', 'weighted_rank']].head(1))

        if not location_only_one_side.empty:
            print("\n📍 Best pair with '01_Location' on one side:")
            print(location_only_one_side[['scan1_path', 'scan2_path', 'weighted_rank']].head(1))

    def determine_goal_based_weights(self, algorithms):
        ##['icp_rank', 'knn_rank', 'iso_rank', 'dbscan_rank']
        """Automatically calculates goal-based normalized weights for ranking features."""
        if algorithms is None:
            algorithms = ['icp_rank', 'knn_rank', 'iso_rank', 'dbscan_rank']
        ranked_results = self.load_ranked_data()
        if not ranked_results:
            logging.error("No data for calculating the weights.")
            return

        df = pd.DataFrame([r.to_dict() for r in ranked_results])
        target_mask = df['scan1_path'].str.contains(self.TARGET_PATTERN) & df['scan2_path'].str.contains(
            self.TARGET_PATTERN)
        weight_scores = {}
        total_score = 0



        for col in algorithms:
            max_val = df[col].max() if df[col].max() != 0 else 1
            df[f'{col}_norm'] = df[col] / max_val
            goal_mean = df.loc[target_mask, f'{col}_norm'].mean()
            score = 1 - goal_mean  # niedriger besser
            weight_scores[col] = score
            total_score += score

        weights = {col: val / total_score for col, val in weight_scores.items()}
        logging.info(f"📊 Automatically determined weights: {weights}")

        return weights

    def consensus_filtering(self, out_csv="consensus_filter_export"):
        ranked_results = self.load_ranked_data()
        df = pd.DataFrame([r.to_dict() for r in ranked_results])
        df['is_consensus'] = (
            (df['dbscan_rank'] >= df['dbscan_rank'].max() - 5) &
            (df['iso_rank'] >= df['iso_rank'].max() - 20) &
            ((df['icp_rank'] + df['knn_rank']) > 100)
        )
        algos_as_string = "algo"
        for a in algorithms:
            if a in df.columns:
                algos_as_string += f"_{a}"
        out_csv = out_csv +"_algo_"+algos_as_string + ".csv"
        df.to_csv(out_csv, index=False)
        print(f"Consensus results saved in {out_csv}")

    def getPercentage(self):
        pass


if __name__ == "__main__":
    #If you let all algorithms in the list, it will run all combinations of 2 to n algorithms
    #You need at least 2 algorithms to form a combination
    algorithms = ['icp_rank', 'knn_rank', 'iso_rank', 'dbscan_rank']
    all_combinations = []
    for r in range(2, len(algorithms) + 1):
        combos = list(itertools.combinations(algorithms, r))
        all_combinations.extend(combos)

    # Loop through all combinations
    for combo in all_combinations:
        print(f"Combination: {combo}")
        pipeline = AnalysisPipeline()
        print("Starting analysis...")
        pipeline.analyze_results(combo)
        pipeline.weighted_aggregation(combo)

        print("Running consensus filtering...")
        pipeline.consensus_filtering()

        print("Searching for unique DBSCAN findings...")
        pipeline.getPercentage()
        print("Process completed!")
