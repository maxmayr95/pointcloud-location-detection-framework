import os
from datetime import datetime


class ICPResult:
    """Stores ICP comparison results between two scans."""

    def __init__(self, dataset_name1, sub_dataset_name1, dataset_name2, sub_dataset_name2, scan1_path, scan2_path,
                 residual_error):
        self.dataset_name1 = dataset_name1
        self.sub_dataset_name1 = sub_dataset_name1
        self.dataset_name2 = dataset_name2
        self.sub_dataset_name2 = sub_dataset_name2
        self.scan1_path = scan1_path
        self.scan2_path = scan2_path
        self.residual_error = residual_error

    def to_dict(self):
        """Convert the ICPResult to a dictionary for MongoDB insertion."""
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
            "timestamp": datetime.now()
        }

    def __str__(self):
        scan1_name = os.path.basename(self.scan1_path)
        scan2_name = os.path.basename(self.scan2_path)
        if self.dataset_name1 == self.dataset_name2 and self.sub_dataset_name1 == self.sub_dataset_name2:
            return f"Dataset: {self.dataset_name1} | Sub-dataset: {self.sub_dataset_name1} | Scan 1: {scan1_name}, Scan 2: {scan2_name}, Residual Error: {self.residual_error:.6f}"
        else:
            return f"Dataset: {self.dataset_name1} | Scan 1: {self.sub_dataset_name1}/{scan1_name}, Scan 2: {self.sub_dataset_name2}/{scan2_name}, Residual Error: {self.residual_error:.6f}"
