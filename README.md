# Robust Verification of Consumer-Grade LiDAR Point Clouds
### Hybrid Algorithm Evaluation for Urban Scene Comparison

---

## 📖 Overview

This repository contains the codebase for the master's thesis:

> **Robust Verification of Consumer-Grade LiDAR Point Clouds: Hybrid Algorithm Evaluation for Urban Scene Comparison**

The framework evaluates the similarity and anomalies between LiDAR scan pairs using three core algorithms:

- **ICP (Iterative Closest Point)** – Geometric alignment and residual error calculation  
- **Isolation Forest** – Anomaly detection on point cloud distributions
- **HDBSCAN (Hierarchical Density-Based Spatial Clustering)** – Clustering-based comparison and noise detection in point clouds
- **KNN Analysis** – Local density comparison relative to a location baseline  

All results are ranked and stored in a MongoDB database for further evaluation.


---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/maxmayr95/pointcloud-location-detection-framework.git
   cd pointcloud-location-detection-framework
    ```
2. Create a virtual environment (optional but recommended):
    ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`
      ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## 🗄️ MongoDB Setup

This project requires a running **MongoDB instance** on `localhost:27017`.  
The database (`anomaly_detection`) and collections are **created automatically** when you run the pipeline.

### Run MongoDB via Docker (Recommended)

```bash
# Pull and run the latest MongoDB container
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -v mongodb_data:/data/db \
  mongo:latest
```
## 📂 How to Use

1️⃣ **Add your dataset**

- Place your dataset inside the `dataset/` folder.
- Follow the same folder structure as shown in the `demo_dataset/` example:
    ```
    dataset/
    └── demo_dataset/
        └── 01_Location/
            ├── scan_1.las
            └── scan_2.las
        └── 02_Location/
            ├── scan_1.las
            └── scan_2.las
    ```
- The **first-level folder must be named `01_Location`**.
- You can add as many scans per location as you want and as many locations as needed.

---

2️⃣ **Execution time warning**

- The pipeline compares all scans **pairwise**, resulting in **O(n²) complexity**:
- Example: 10 scans → 45 comparisons
- Example: 100 scans → 4950 comparisons
- More scans = more execution time.
- For new incoming scans, you can optimize by comparing only against a **subset** of existing scans.

3️⃣  **Run the pipeline**
```bash
main() in pipeline.py 
```
This will process all scans in the `dataset/` folder and store results in MongoDB.

4️⃣ **Run the analysis file**

There is a file called `analysis.py` that contains the main function to run the analysis.
It analyses the results from the MongoDB database and prints the results to the console.
It will show you how efficient the certain algorithms are and how many anomalies were detected.

5️⃣ **Run weighted aggregation and advanced analysis (optional)**

You can further evaluate the detection capabilities of the different algorithms by performing weighted aggregation analysis. This approach tests all possible combinations of anomaly detection methods (ICP, Isolation Forest, HDBSCAN, KNN) to find out which combinations—and which weights—work best for identifying target pairs (such as `"01_Location"`).

- For each combination (pairs, triples, all four algorithms), the pipeline:
    - **Aggregates multiple rankings into one weighted score** (automation determines optimal weights for your anomaly target).
    - **Ranks all anomaly pairs** according to the new aggregated score.
    - **Outputs detailed statistics** and **exports per-combination CSVs** for further review.

```bash
run the file: weighted_aggregation_dynamic_combinations.py
```

- This script will:
    - Automatically run the aggregation for each relevant combination of algorithms.
    - Print summaries, rank positions, and statistics for each loop.
    - Save detailed CSV files with ranked results for every tested algorithm combination.

**When to use this:**
> Use this analysis to determine which combinations and weightings of algorithms most reliably surface your target pairs at the top—helping you optimize your anomaly detection strategy for future LiDAR datasets.



## 📊 Results
The results are stored in the MongoDB database under the `anomaly_detection` database and `results` collection.
You can query the results using MongoDB Compass or any MongoDB client.


## ⚙️ Requirements

- Python **3.9+**
- MongoDB (local or remote)
- Recommended: ≥8 CPU cores for parallel processing

Install dependencies:
```bash
pip install -r requirements.txt
```
## 📜 License

This project is licensed under the **MIT License** – you are free to use, modify, and distribute it, provided that proper credit is given.

See the [LICENSE](LICENSE) file for details.