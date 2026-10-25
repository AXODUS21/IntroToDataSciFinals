# Seoul Bike Sharing Demand Prediction
## Supervised Binary Classification Pipeline
**De La Salle Lipa**  
College of Information Technology and Engineering  
Department of Computer Science  

---

### Project Information

* **Course:** Introduction to Data Science (Intdsci)
* **Section:** C2B
* **Instructor:** Ms. Donna Garcia
* **Group:** FML
* **Members:**
  * Fabello, Courtney Judge
  * Lubi, Axellerosh
  * Macaraig, Paolo Adrian
* **Date of Submission:** Friday, October 30, 2026
* **Dataset:** Seoul Bike Sharing Demand Dataset (UCI Machine Learning Repository)
* **Task:** Supervised Binary Classification (`is_rented`: 0 = Not Rented, 1 = Rented)

---

### Project Overview

This project develops an end-to-end machine learning pipeline to predict whether public bicycle rentals occur during any given hour in Seoul, South Korea. The target variable is formulated as a binary classification problem:
* **Class 0 (Not Rented):** `Rented Bike Count == 0` (3.37% of observations)
* **Class 1 (Rented):** `Rented Bike Count > 0` (96.63% of observations)

Predictors comprise 13 meteorological and temporal attributes, including temperature, humidity, wind speed, solar radiation, rainfall, snowfall, hour of day, and calendar indicators.

#### Key Methodological Steps
1. **Data Cleaning & Audit:** Verified complete data integrity (0 null values, 0 duplicate rows) across all 8,760 hourly observations.
2. **Data Leakage Prevention:** Audited the `Functioning Day` feature, which showed a 100% deterministic link to zero rental hours (all 295 zero-rental cases occurred strictly when `Functioning Day == 'No'`). The column was removed from predictor matrix $X$ to prevent artificial test leakage.
3. **Multicollinearity Remediation:** Identified an $r = 0.9128$ correlation between temperature and dew point temperature; dew point temperature was dropped.
4. **Reproducible Partitioning:** Employed an 80-20 stratified train-test split locked with `random_state=42` (7,008 training samples, 1,752 testing samples).
5. **Leak-Free Preprocessing:** Standardized numeric features and one-hot encoded categorical variables using a Scikit-Learn `ColumnTransformer` fitted strictly on `X_train`.
6. **Class Imbalance Handling:** Implemented balanced class weighting ($w_j = \frac{N}{2 \times N_j}$) across candidate classifiers.

---

### Model Evaluation Results

Models were trained and evaluated on the untouched 20% test set ($N = 1,752$):

| Model | Accuracy | Balanced Accuracy | Recall (Class 0) | Precision (Class 0) | Macro F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Dummy - Majority) | 96.63% | 50.00% | 0.00% | 0.00% | 0.4914 | 0.5000 |
| Logistic Regression (Balanced) | 73.00% | 83.58% | 94.92% | 10.65% | 0.5147 | 0.8868 |
| Decision Tree (Balanced) | 80.37% | 89.84% | 100.00% | 14.64% | 0.5712 | 0.9180 |
| Random Forest (Balanced) | **84.65%** | **91.24%** | **98.31%** | **17.79%** | **0.6075** | **0.9664** |

Due to the extreme 96.6% / 3.4% class imbalance, raw accuracy is uninformative (a naive baseline predicting only the majority class achieves 96.63% accuracy while failing to detect any zero-rental hours). The **Balanced Random Forest** model delivered the strongest overall performance, achieving a **91.24% balanced accuracy** and an **ROC-AUC of 0.9664**, successfully detecting 58 out of 59 non-rented hours in the test set.

---

### Repository Structure

```
IntroToDataSciFinals/
├── run_pipeline.py                        # Standalone end-to-end Python pipeline
├── requirements.txt                       # Python dependencies
├── README.md                              # Project documentation and guide
│
├── data/
│   └── SeoulBikeData.csv                  # Dataset (8,760 records)
│
├── figures/                               # Generated 300 DPI evaluation charts
│   ├── 01_target_distribution.png         # Target class distribution
│   ├── 02_hourly_seasonal_rentals.png     # Average hourly rentals across seasons
│   ├── 03_weather_distributions.png       # Weather variable boxplots by class
│   ├── 04_correlation_matrix.png          # Correlation matrix heatmap
│   ├── 05_confusion_matrices.png          # Test set confusion matrices
│   ├── 06_roc_pr_curves.png               # ROC curves
│   └── 07_feature_importance.png          # Random Forest Gini feature importances
│
├── notebooks/
│   └── final_project.ipynb                # Executed Jupyter Notebook
│
├── reports/
│   ├── FINAL_PROJECT_DOCUMENTATION.docx   # Word format project report
│   ├── FINAL_PROJECT_DOCUMENTATION.md     # Markdown project documentation
│   └── pipeline_metrics.json              # Serialized model metrics
│
└── scripts/
    ├── build_notebook.py                  # Script to build and execute notebook
    └── generate_docx.py                   # Script to build Word document
```

---

### Setup and Execution

#### Environment Setup
Ensure Python 3.10 or higher is installed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Run the Machine Learning Pipeline
To run data ingestion, cleaning, EDA figure generation, cross-validation, and test evaluation:

```powershell
python run_pipeline.py
```

#### Generate the Word Report
To compile the markdown documentation and embedded figures into `reports/FINAL_PROJECT_DOCUMENTATION.docx`:

```powershell
python scripts/generate_docx.py
```

#### Re-Execute the Jupyter Notebook
To execute all notebook cells fresh and embed outputs into `notebooks/final_project.ipynb`:

```powershell
python scripts/build_notebook.py
```

---

### References
* Sathishkumar, V. E., Park, J., & Cho, Y. (2020). Using data mining techniques for bike sharing demand prediction in metropolitan city. *Computer Communications*, 153, 353–366. https://doi.org/10.1016/j.comcom.2020.02.007
* UCI Machine Learning Repository: Seoul Bike Sharing Demand Dataset.
