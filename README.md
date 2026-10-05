# CMPE 255 — Clustering and AutoML

A six-part Google Colab notebook series for **CMPE 255 Data Mining**, covering K-means clustering from first principles through modern AutoML and MLOps workflows with **AutoGluon**, **NVIDIA RAPIDS**, and **PyCaret**. Each notebook pairs algorithm-focused study with hands-on demonstrations, preserves the outputs of completed runs, and is accompanied by a narrated video walkthrough.

## The six parts

| Part | Notebook | Focus | Walkthrough |
|------|----------|-------|-------------|
| 1 | `01_kmeans.ipynb` | K-means clustering and variations | [YouTube](https://youtu.be/T8vqyqKRv-8) |
| 2 | `02_autogluon_capabilities.ipynb` | AutoGluon capabilities landscape | [YouTube](https://youtu.be/l2RCSJbBUqw) |
| 3 | `03_autogluon_end_to_end.ipynb` | AutoGluon end-to-end ML workflow | [YouTube](https://youtu.be/YK-V9t7ZHz0) |
| 4 | `04_rapids_cpu_vs_gpu.ipynb` | NVIDIA RAPIDS and CPU/GPU comparison | [YouTube](https://youtu.be/vwsBTDRIa_M) |
| 5 | `05_pycaret_capabilities.ipynb` | PyCaret capabilities landscape | [YouTube](https://youtu.be/s09tZyYAld4) |
| 6 | `06_pycaret_mlops.ipynb` | PyCaret MLOps workflow | [YouTube](https://youtu.be/bh9DQB_q2nQ) |

## What each part covers

### Part 1 — K-means clustering and variations
- The K-means objective (SSE) and why the centroid is the mean — derived and verified symbolically
- Lloyd's algorithm implemented from scratch in NumPy, watched frame by frame, with a convergence proof and numerical equivalence check against scikit-learn
- Initialization: the random-init trap, `n_init`, K-means++ with its O(log K) guarantee, empty clusters, and outliers
- Bisecting K-means from scratch, plus a Lloyd-vs-bisecting comparison on SSE, time, and accuracy
- Variants from swapping the objective: K-medians, spherical K-means, K-medoids, mini-batch, fuzzy c-means, GMM/EM, kernel K-means
- Validation with internal (silhouette, Calinski–Harabasz, Davies–Bouldin) and external (purity, ARI, NMI) metrics; choosing K via elbow, gap statistic, and stability
- Failure cases (sizes, densities, shapes) and remedies; K-means vs hierarchical vs DBSCAN
- End-to-end applications: customer segmentation, color quantization, digit clustering; MLOps (pipelines, drift monitoring); embeddings and the 2026 frontier (GPU K-means, vector quantization, LLM-named clusters)
- Seven Chapter-7 textbook exercises, each verified in code

### Part 2 — AutoGluon capabilities landscape
- Binary classification (telecom churn), multiclass classification (loan grades), and regression (house prices) from the same three-line pattern
- Quantile regression producing delivery-time windows, with coverage checked against the target
- Rare-event fraud detection with cost-aware threshold selection
- Time-series forecasting, including a zero-shot Chronos foundation model compared against trained forecasters
- Text + tabular (TF-IDF vs sentence embeddings), image classification (defect photos), and embedding-based semantic search over support tickets
- Tabular foundation models vs gradient boosting: the accuracy/latency trade-off, measured
- Shipping: persisted pipelines, slim deployment clones, inference latency, and drift monitoring

### Part 3 — AutoGluon end-to-end ML workflow
- Data inspection, hand-built baselines, leaderboards, and the metric families — including which metric lies and when
- Class imbalance, decision thresholds as business decisions, and probability calibration
- Ensembles and stacking with out-of-fold predictions; hyperparameter search with validation-vs-test rank checks
- Feature importance, per-prediction explanations, and partial dependence surfaces
- The leakage trap caught and diagnosed; presets and time budgets compared; text-column ablations
- Persistence, monitoring with PSI, error analysis, and a diagnostics checklist

### Part 4 — NVIDIA RAPIDS and CPU/GPU comparison
- Why GPUs are fast: columnar layout, and honest benchmarking — warm vs first call, always counting the host-to-device copy
- cuDF as pandas on the GPU: verbs, strings, dates, missing values, and a probe-by-probe compatibility check
- `cudf.pandas` zero-code-change mode with the profiler showing which calls ran on GPU and which silently fell back
- Exploratory analysis at speed; aggregate-on-GPU / plot-on-host discipline; the small-data crossover where pandas wins
- Preprocessing for cuML; float32 vs float64 measured; the cuML model zoo through the scikit-learn API
- Regression, GPU-resident pipelines, k-means / DBSCAN / HDBSCAN, PCA / UMAP / t-SNE, nearest neighbors, anomaly detection
- Cross-validation and grid search over cuML estimators; `cuml.accel` zero-code acceleration
- Memory management: dtype shrinking, the RMM pool, spilling, a deliberate out-of-memory error, chunked Parquet, and Dask-cuDF
- XGBoost on GPU with early stopping and GPU SHAP; graph analytics; model saving, serving latency, and drift
- Where CPUs still win: the 100k-row crossover, single-row serving, and the cost math

### Part 5 — PyCaret capabilities landscape
- Binary and multiclass classification and regression from `setup` → `compare_models` → `predict_model`
- Target transforms for skewed regression targets; tuning, bagging, boosting, blending, stacking, calibration, and threshold optimization
- Rare-event modeling with SMOTE rebalancing and dollar-priced threshold decisions
- Clustering (with the one-hot-hijack warning) and anomaly detection with a contamination-fraction caveat
- Time series with exogenous variables: measuring what the promo column is actually worth
- Text in tables via three routes — raw category, built-in TF-IDF, hand-engineered features — compared head to head
- SHAP explanations, reason codes, and fairness checks; deployment via `finalize_model`, saved pipelines, generated REST APIs, PSI drift alarms, and the retrain loop

### Part 6 — PyCaret MLOps workflow
- A hand-built scikit-learn baseline before any low-code, then all 73 `setup` knobs tried one at a time against a reference run
- Pipeline internals: split hygiene, imputation learned from training only, and two leaks hiding in the split itself
- Model comparison, metric families, and the full plot gallery; tuning, ensembles, calibration, and cost-based thresholds
- Regression with residual analysis; the leakage trap (R² ≈ 1.0 vs honest R²) diagnosed via feature importance
- SHAP with additivity checks, permutation importance, partial dependence, and interaction discovery
- Clustering and anomaly detection on spending behavior; time series where the naive model wins on 24 points; per-series vs panel forecasting
- `finalize_model` (never report metrics after it), save/load round trips, API + Dockerfile generation, MLflow logging, fairness gaps, Evidently drift reports, and the retrain recipe
- Error analysis with confident-mistake profiling, a 12-point diagnostics checklist, and an honest 2026 verdict on where low-code fits

## Running and reproducibility

The notebooks are intended to be opened and executed in Google Colab from top to bottom. Run the setup and dependency cells first, then the sections in order. The RAPIDS notebook requires a GPU-enabled Colab runtime.

The committed `.ipynb` files retain the outputs generated during execution — printed results, tables, plots, metrics, and other artifacts. Runtime and numerical results can vary with the Colab runtime, library versions, available hardware, and execution conditions.

For the most reliable rerun: open each notebook in Colab, select the appropriate runtime, restart the session when required by the setup cells, and run all cells sequentially.

## Walkthrough videos

The assignment requires a detailed walkthrough of the six notebooks, explaining the important code sections, what the major cells do, and the outputs produced during the student's own Colab execution. Per-part walkthroughs are linked in the table above.

Tutorial video: _Add the final video link here._

## Repository structure

```
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

CMPE 255 — Data Mining
