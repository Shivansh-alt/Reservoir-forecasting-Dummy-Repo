# Weekly Reservoir Storage Forecasting

This repository contains data and machine learning models for forecasting weekly reservoir storage levels in India, supporting drought preparedness (SDG 6 & 13).

## Repository Structure
- `reservoir_dataset.csv`: Extracted sample dataset of Central Water Commission (CWC) reservoir levels.
- `forecasting_models.py`: Python script containing feature engineering (lag generation) and regression model pipelines.
- `requirements.txt`: Python dependencies required to run the code.

## Models Implemented
- Linear Regression (with lag features)
- Polynomial Regression (Degree 2)
- Ridge Regression (L2 Regularization)
- k-Nearest Neighbors (k-NN) Regressor

## Metrics Evaluated
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared ($R^2$)

## Usage
1. Clone the repository and navigate into the directory.
2. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the training script:
   ```bash
   python forecasting_models.py
   ```
