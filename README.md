# Medical Cost and Heart Disease Prediction

Machine learning project built in Python with scikit-learn, covering regression, time-series validation and classification on real-world datasets. The focus is on healthcare and insurance data: predicting medical insurance charges and classifying heart disease risk.

## What's inside

| Part | Task | Models | Key result |
|---|---|---|---|
| A | Predict medical insurance charges from age, BMI, children, sex, smoking status and region | Linear regression vs ridge regression | R² 0.78 (MAE about 4,181) |
| B | Forecast daily and hourly bike rental demand, validated in date order to avoid using future data | Random forest regressor | Daily MAE 1,040 on mean demand of 5,898 |
| C | Classify Titanic survival and breast cancer diagnosis, and study how the decision threshold changes results | Logistic regression vs random forest | Breast cancer ROC AUC 0.996 |
| D | Tune a model with 5-fold cross-validation, then test once on held-out data | Logistic regression + GridSearchCV | Test ROC AUC 0.837 |
| E | Save models with joblib and use them on new data | Saved pipelines + model card | Smoker vs non-smoker with the same profile: predicted charges about 4.6x higher |
| Extra | Heart disease classification on 918 patient records | Logistic regression vs random forest | Random forest: 90.4% accuracy, ROC AUC 0.944 |

## Highlights

- **Insurance cost drivers:** for the same customer profile, the regression model predicts charges of about 30,177 for a smoker versus 6,526 for a non-smoker.
- **Heart disease threshold analysis:** compared decision thresholds of 0.5, 0.4 and 0.3 to balance missed patients (false negatives) against false alarms (false positives), which matters more than raw accuracy in a medical setting.
- **Good practice:** preprocessing pipelines (imputation, scaling, one-hot encoding), stratified splits, chronological validation for time series, cross-validated tuning and saved models with a model card.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, joblib, Jupyter

## How to run

1. Install the packages: `pip install -r requirements.txt`
2. Download the datasets listed in `data/README.md` and save them in a folder called `all_datasets` one level above this project folder (the notebook reads from `../all_datasets`).
3. Open `medical_cost_heart_disease_ml.ipynb` in Jupyter or VS Code and run all cells.

## Author

Adrianna Sofea, Bachelor of Computer Science (Hons), Data Analytics, Universiti Teknologi PETRONAS
