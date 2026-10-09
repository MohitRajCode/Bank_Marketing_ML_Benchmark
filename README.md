# Bank Marketing: Random Forest vs XGBoost

A machine learning classification project comparing Random Forest and XGBoost for predicting whether a bank customer subscribes to a term deposit.

## Models

- **Random Forest:** 100 decision trees with `max_depth=10`.
- **XGBoost:** Gradient-boosted decision trees with validation-based early stopping.

## Workflow

1. Load and inspect the Bank Marketing dataset.
2. Separate input features and the target variable.
3. Split the data into training, validation, and test sets.
4. Apply one-hot encoding to categorical features.
5. Train both models and measure training time.
6. Compare test-set accuracy and other classification metrics.
7. Visualize feature importance side by side.

## Evaluation

The models are compared using their measured test-set performance, training time, and feature-importance rankings.

Actual results are reported after running the final evaluation.

## Technologies

Python, Pandas, Matplotlib, Scikit-learn, XGBoost, and KaggleHub.
