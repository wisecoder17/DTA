# Delivery ETA System Guide

This guide provides step-by-step instructions on how to operate the Delivery ETA system. Follow the steps below to process data, perform exploratory data analysis (EDA), train models, make predictions, and evaluate the results.

---

## 1. Data Processing

### Script: `src/preprocessing.py`

1. **Purpose**: This script processes raw data and prepares it for analysis and modeling.
2. **Input**: Raw data files located in `data/raw/`.
3. **Output**: Processed data saved in `data/processed/`.
4. **How to Run**:
   ```bash
   python src/preprocessing.py
   ```

---

## 2. Exploratory Data Analysis (EDA)

### Script: `src/eda_summary.py`

1. **Purpose**: This script generates a summary of the dataset, including key statistics and visualizations.
2. **Input**: Processed data from `data/processed/`.
3. **Output**: EDA results saved in `results/` (e.g., `eda_summary.csv`) and visualizations in `results/figures/`.
4. **How to Run**:
   ```bash
   python src/eda_summary.py
   ```

---

## 3. Model Training

### Script: `src/train_model.py`

1. **Purpose**: This script trains machine learning models using the processed data.
2. **Input**: Processed data from `data/processed/`.
3. **Output**: Trained model artifacts saved in `models/`.
4. **How to Run**:
   ```bash
   python src/train_model.py
   ```

---

## 4. Making Predictions

### Script: `src/predict.py`

1. **Purpose**: This script uses the trained model to make predictions on new data.
2. **Input**: New data or test data.
3. **Output**: Predictions saved in `results/` (e.g., `amazon_model_results.csv`).
4. **How to Run**:
   ```bash
   python src/predict.py
   ```

---

## 5. Model Evaluation

### Script: `src/evaluate_model.py`

1. **Purpose**: This script evaluates the performance of the trained model using metrics such as accuracy, precision, recall, etc.
2. **Input**: Predictions and ground truth data.
3. **Output**: Evaluation metrics saved in `results/tables/` (e.g., `amazon_test_metrics.csv`).
4. **How to Run**:
   ```bash
   python src/evaluate_model.py
   ```

---

## 6. Deep Learning Model

### Script: `src/dl_model.py`

1. **Purpose**: This script contains a deep learning model built from scratch for advanced predictions.
2. **How to Run**:
   ```bash
   python src/dl_model.py
   ```

---

## Notes

- Ensure all dependencies are installed before running the scripts. Use the `requirements.txt` file to install them:
  ```bash
  pip install -r requirements.txt
  ```
- Follow the sequence of operations as outlined above for optimal results.
- Refer to the `notebooks/` directory for additional exploratory analysis and modeling insights.

---

