# Cancer Classification Assignment

## Overview
This Assignment implements a cancer classification system using Random Forest with feature selection and data augmentation. The classifier processes gene expression data to predict cancer labels, focusing on achieving high ROC AUC scores.

## Key Components
- **Feature Selection**: Selects the most predictive genes using SelectKBest
- **Data Augmentation**: Improves model robustness by adding controlled Gaussian noise
- **Random Forest Classifier**: Ensemble learning method for classification

## Functions

### `__init__()`
Initializes the CancerClassifier with paths for training, test, and output files. It loads the data and prepares the model components.

### `load_data()`
Loads and preprocesses training and test datasets from CSV files.

### `augment_with_noise()`
Creates augmented samples by adding Gaussian noise to the original data. This helps improve model generalization by artificially increasing the dataset size.

### `select_features()`
Selects the k most informative features (genes) using SelectKBest method, reducing dimensionality and improving model performance.

### `train_model()`
Trains the Random Forest model on a portion of the data and evaluates performance using ROC AUC. Includes optional data augmentation.

### `retrain_on_full_dataset()`
Retrains the model on the complete dataset after validation, preparing for final predictions.

### `predict()`
Generates predictions on the test data and saves results to a CSV file.

## Installing Required Libraries

Ensure you have the necessary dependencies installed before running the script:
```bash
pip install numpy pandas scikit-learn
```

## Running the Assignment
Run the script from the command line:
```bash
python cancer_classification.py --train kaggle_train.csv --test kaggle_test.csv --output submission.csv
```