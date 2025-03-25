# Peptide Classification Assignment

## Overview
This Assignment implements a peptide sequence classification system using an ensemble of machine learning models, including:
- **Random Forest (RF)**
- **Gradient Boosting Classifier (GBC)**
- **Deep Neural Network (DNN)**
- **XGBoost Random Forest (XGBRF)**

The classifier extracts biochemical properties from peptide sequences, processes them into numerical features, and applies machine learning models to predict labels.

## Functions

### `__init__()`
Initializes the PeptideClassifier with paths for training, test, properties, and output files. It also loads the data and initializes models and preprocessing components.

### `load_data()`
Loads and preprocesses training and test datasets. Removes missing values, renames columns for consistency, and converts labels.

### `extract_features(sequences)`
Extracts biochemical properties from peptide sequences and computes numerical features such as molecular weight, hydrophobicity, charge, and amino acid composition.

### `add_gaussian_noise(X, noise_level=0.01)`
Adds Gaussian noise to the training data to improve model generalization.

### `train_models()`
Trains multiple classification models using the extracted features. Implements SMOTE for handling class imbalance, applies feature scaling, and trains the following models:
- Random Forest
- Gradient Boosting Classifier
- XGBoost Random Forest
- Deep Neural Network (DNN) with batch normalization and dropout layers

### `predict()`
Generates predictions using the trained models. Uses an ensemble method with weighted probabilities from different classifiers to produce final predictions and saves them to a file.

## Installing Required Libraries

Ensure you have the necessary dependencies installed before running the script:
```bash
pip install numpy pandas scikit-learn tensorflow imblearn xgboost
```

## Running the Assignment
Run the script from the command line:
```bash
python peptide_classification.py --train train.csv --test test.csv --properties peptide_properties.json --output submission.csv
```

#### Reference for Feature Extraction:- https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/IMGTclasses.html