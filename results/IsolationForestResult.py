import os
from datetime import datetime

class IsolationForestResult:
    """Stores Isolation Forest results for a scan pair."""

    def __init__(self, dataset_name1, sub_dataset_name1, scan1_path,
                 dataset_name2, sub_dataset_name2, scan2_path, iso_score):
        self.dataset_name1 = dataset_name1
        self.sub_dataset_name1 = sub_dataset_name1
        self.scan1_path = scan1_path
        self.dataset_name2 = dataset_name2
        self.sub_dataset_name2 = sub_dataset_name2
        self.scan2_path = scan2_path
        self.iso_score = iso_score

    def to_dict(self):
        """Convert the IsolationForestResult to a dictionary for MongoDB insertion."""
        return {
            "dataset_name1": self.dataset_name1,
            "sub_dataset_name1": self.sub_dataset_name1,
            "dataset_name2": self.dataset_name2,
            "sub_dataset_name2": self.sub_dataset_name2,
            "scan1_path": self.scan1_path,
            "scan2_path": self.scan2_path,
            "scan_name1": os.path.basename(self.scan1_path),
            "scan_name2": os.path.basename(self.scan2_path),
            "iso_score": self.iso_score,
            "timestamp": datetime.now()
        }

    def __str__(self):
        scan1_name = os.path.basename(self.scan_path1)
        scan2_name = os.path.basename(self.scan_path2)
        return (f"Dataset1: {self.dataset_name1} | Sub-dataset1: {self.sub_dataset_name1} | Scan1: {scan1_name}, "
                f"Dataset2: {self.dataset_name2} | Sub-dataset2: {self.sub_dataset_name2} | Scan2: {scan2_name}, "
                f"Isolation Score: {self.iso_score:.6f}")
