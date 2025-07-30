import os
from datetime import datetime


class RankedResult:
    """Stores ranked ICP comparison results between two scans."""

    def __init__(self, dataset_name1, sub_dataset_name1, dataset_name2, sub_dataset_name2, scan1_path, scan2_path,
                 residual_error=None,iso_score=None, total_rank=None, icp_rank=None, dbscan_rank=None, knn_rank=None,iso_rank=None):
        self.dataset_name1 = dataset_name1
        self.sub_dataset_name1 = sub_dataset_name1
        self.dataset_name2 = dataset_name2
        self.sub_dataset_name2 = sub_dataset_name2
        self.scan1_path = scan1_path
        self.scan2_path = scan2_path
        self.residual_error = residual_error
        self.iso_score = iso_score
        self.total_rank = total_rank
        self.icp_rank = icp_rank
        self.dbscan_rank = dbscan_rank
        self.knn_rank = knn_rank
        self.iso_rank = iso_rank

    def to_dict(self):
        """Convert the RankedResult to a dictionary for MongoDB insertion."""
        return {
            "dataset_name1": self.dataset_name1,
            "sub_dataset_name1": self.sub_dataset_name1,
            "dataset_name2": self.dataset_name2,
            "sub_dataset_name2": self.sub_dataset_name2,
            "scan1_path": self.scan1_path,
            "scan2_path": self.scan2_path,
            "scan1_name": os.path.basename(self.scan1_path),
            "scan2_name": os.path.basename(self.scan2_path),
            "residual_error": self.residual_error,
            "iso_score": self.iso_score,
            "total_rank": self.total_rank,
            "icp_rank": self.icp_rank,
            "dbscan_rank": self.dbscan_rank,
            "knn_rank": self.knn_rank,
            "iso_rank": self.iso_rank,
            "timestamp": datetime.now()
        }

    def __str__(self):
        scan1_name = os.path.basename(self.scan1_path)
        scan2_name = os.path.basename(self.scan2_path)
        rank_info = f"Total Rank: {self.total_rank} | ICP Rank: {self.icp_rank} | DBSCAN Rank: {self.dbscan_rank} | KNN Rank: {self.knn_rank} | ISO Rank: {self.iso_rank}"
        return f"{rank_info} | Dataset: {self.dataset_name1} | Sub-dataset: {self.sub_dataset_name1} | Scan 1: {scan1_name}, Scan 2: {scan2_name}, Residual Error: {self.residual_error:.6f}"
