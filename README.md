# CMPE 255 — Clustering and AutoML

This repository contains six Google Colab notebooks for CMPE 255 Data Mining. The notebooks cover K-means clustering, AutoGluon, NVIDIA RAPIDS, PyCaret, and MLOps, with a combination of algorithm-focused study and practical capability demonstrations.

The work is organized into six assignment parts. Each notebook is based on the corresponding instructor-provided Colab and is intended to be executed in the student's own Google Colab environment. The notebooks preserve the outputs of completed runs for review and execution.

## Six parts

| **Part** | **Notebook** | **Focus** | **YouTube** |
| -------- | ------------ | --------- | ----------- |
| 1 | [`01_kmeans.ipynb`](part1-kmeans/01_kmeans.ipynb) | K-means clustering and variations |  |
| 2 | [`02_autogluon_capabilities.ipynb`](part2-autogluon/02_autogluon_capabilities.ipynb) | AutoGluon capabilities landscape |  |
| 3 | [`03_autogluon_end_to_end.ipynb`](part3-autogluon/03_autogluon_end_to_end.ipynb) | AutoGluon end-to-end ML workflow |  |
| 4 | [`04_rapids_cpu_vs_gpu.ipynb`](part4-rapids/04_rapids_cpu_vs_gpu.ipynb) | NVIDIA RAPIDS and CPU/GPU comparison |  |
| 5 | [`05_pycaret_capabilities.ipynb`](part5-pycaret/05_pycaret_capabilities.ipynb) | PyCaret capabilities landscape |  |
| 6 | [`06_pycaret_mlops.ipynb`](part6-pycaret/06_pycaret_mlops.ipynb) | PyCaret MLOps workflow |  |

## What the notebooks cover

### Part 1 — K-means

The K-means notebook develops clustering from the algorithmic foundations through practical variations and evaluation. It covers the K-means objective, Lloyd's algorithm, initialization and K-means++, convergence, empty clusters, outliers, bisecting K-means, alternative clustering objectives, validation metrics, methods for selecting the number of clusters, common failure cases, and practical clustering applications.

### Part 2 — AutoGluon Capabilities

The AutoGluon capabilities notebook provides a broad tour of the framework across multiple machine-learning problem types. It demonstrates classification, regression, quantile prediction, rare-event problems, time-series forecasting, multimodal workflows, text and tabular data, image tasks, embeddings, semantic search, and other AutoGluon capabilities.

### Part 3 — AutoGluon End-to-End

The AutoGluon end-to-end notebook follows a complete AutoML workflow from data inspection and baseline models through model training, leaderboards, evaluation metrics, class imbalance, thresholds, calibration, ensembles, hyperparameter search, feature importance, prediction explanations, forecasting, multimodal workflows, persistence, monitoring, and error analysis.

### Part 4 — NVIDIA RAPIDS

The RAPIDS notebook explores GPU-accelerated data science using NVIDIA RAPIDS and compares GPU-oriented workflows with CPU implementations. It covers GPU setup and benchmarking, cuDF, cuML, preprocessing, clustering, dimensionality reduction, nearest-neighbor workflows, anomaly detection, model search, XGBoost, SHAP, graph analytics, memory management, and practical CPU fallback considerations.

### Part 5 — PyCaret Capabilities

The PyCaret capabilities notebook provides a broad tour of PyCaret's low-code machine-learning workflows. It demonstrates classification, regression, clustering, anomaly detection, time-series forecasting, text-related workflows, model comparison, tuning, ensemble methods, visualization, interpretability, and deployment-oriented capabilities.

### Part 6 — PyCaret MLOps

The PyCaret MLOps notebook follows the machine-learning lifecycle from data inspection and baseline modeling through experiment setup, preprocessing, model comparison, evaluation, tuning, ensembles, calibration, interpretability, clustering, anomaly detection, forecasting, model persistence, prediction, experiment logging, API generation, fairness checks, and data-drift monitoring.

## Running and reproducibility

The notebooks are intended to be opened and executed in Google Colab from top to bottom. Setup and dependency cells should be executed first, followed by the notebook sections in order. The RAPIDS notebook requires a compatible GPU-enabled Colab runtime.

The committed `.ipynb` files are intended to retain the outputs generated during execution, including printed results, tables, plots, metrics, and other notebook artifacts. Runtime and numerical results can vary with the Colab runtime, library versions, available hardware, and execution conditions.

For the most reliable rerun, open each notebook in Colab, select the appropriate runtime, restart the session when required by the setup cells, and run all cells sequentially.

## Walkthrough video

The assignment requires a detailed walkthrough of the six notebooks. The walkthrough explains the important code sections, what the major cells are doing, and the outputs produced during the student's own Colab execution.

**Tutorial video:** Add the final video link here.

## Repository structure

```text
cmpe255-automl-kmeans-rapids-pycaret/
│
├── README.md
│
├── part1-kmeans/
│   └── 01_kmeans.ipynb
│
├── part2-autogluon/
│   └── 02_autogluon_capabilities.ipynb
│
├── part3-autogluon/
│   └── 03_autogluon_end_to_end.ipynb
│
├── part4-rapids/
│   └── 04_rapids_cpu_vs_gpu.ipynb
│
├── part5-pycaret/
│   └── 05_pycaret_capabilities.ipynb
│
└── part6-pycaret/
    └── 06_pycaret_mlops.ipynb
```

## Reference Colabs

The notebooks correspond to the six reference Colabs covering K-means clustering, AutoGluon capabilities, AutoGluon end-to-end ML, NVIDIA RAPIDS, PyCaret capabilities, and PyCaret MLOps.

## Course

**CMPE 255 — Data Mining**

**Repository:** `cmpe255-automl-kmeans-rapids-pycaret`
