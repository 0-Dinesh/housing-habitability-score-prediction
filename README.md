# Housing Habitability Score Prediction Engine

An applied machine learning system leveraging ensemble tree regressors to estimate residential habitability ratings from environmental, structural, and neighborhood attributes.

---

## Performance Benchmark

| Metric | Holdout Test Set Performance |
|---|---|
| **$R^2$ Score** | **0.8319 (83.2% variance explained)** |
| **Mean Absolute Error (MAE)** | **4.5010 points** |
| **Total Processed Samples** | 35,549 records |
| **Features Extracted** | 25 engineered dimensions |

---

## Data Pipeline & Architecture

1. **Missing Data Imputation:**
   - Continuous attributes (`Number_of_Windows`, `Frequency_of_Powercuts`) imputed using median heuristics.
   - Categorical indicators (`Furnishing`, `Crime_Rate`, `Dust_and_Noise`) imputed via statistical mode.
2. **Feature Engineering:**
   - Stripped arbitrary identifiers (`Property_ID`).
   - One-hot encoded categorical series (Property Type, Furnishing Status, Power Backup, Water Supply schedules, Crime Rate, and Noise levels) with baseline reference drop to prevent multi-collinearity.
3. **Model Selection:**
   - Trained a 100-estimator `RandomForestRegressor` with multi-core parallelization (`n_jobs=-1`).
4. **Serialization:**
   - Persisted engineered columns, train/test subsets, and model estimators using `joblib`.

---

## Project Structure

```text
housing-habitability-score-prediction/
├── data/
│   └── Housing_Habitability_Dataset.csv
├── models/
│   ├── Dataset_ColumnsAlone.joblib
│   ├── Test_Dataset.joblib
│   └── Trained_HH_Model.joblib
├── notebooks/
│   ├── 01_data_exploration_and_training.ipynb
│   └── 02_model_evaluation.ipynb
├── .gitignore
├── LICENSE
├── housing-habitability-score-prediction-report.pdf
├── README.md
└── requirements.txt
```

---

## Getting Started

### Prerequisites
- Python 3.9+

### Environment Setup

1. Clone repository:
   ```bash
   git clone https://github.com/your-username/housing-habitability-score-prediction.git
   cd housing-habitability-score-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution

Open the interactive notebooks via Jupyter Lab or VS Code:
- Run `notebooks/01_data_exploration_and_training.ipynb` to clean data and train the model.
- Run `notebooks/02_model_evaluation.ipynb` to compute model accuracy benchmarks on the test set.

---

## License

Distributed under the [MIT License](LICENSE).