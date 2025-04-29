import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class CancerClassifier:
    '''
    A class to classify cancer samples using gene expression data.
    Utilizes Random Forest with data augmentation and feature selection.
    '''

    def __init__(self, train_path, test_path, output_path):
        '''
        Initializes the CancerClassifier with paths for training, test, and output.
        '''

        self.selector = None
        self.classifier = None
        self.test_path = test_path
        self.train_path = train_path
        self.output_path = output_path

        self.load_data()

    def load_data(self):
        '''
        Loads and preprocesses training and test datasets.
        '''

        self.train_data = pd.read_csv(self.train_path, header=None)
        self.test_data  = pd.read_csv(self.test_path , header=None)
        self.X = self.train_data.iloc[:, 1:].values
        self.y = self.train_data.iloc[:,  0].values

    def augment_with_noise(self, X, y, noise_factor=0.1, num_copies=3):
        '''
        Augments the dataset by adding Gaussian noise to each sample.
        '''

        if len(X) == 0:
            return X, y

        feature_std = np.std(X, axis=0)
        X_augmented = [x.copy() for x in X]
        y_augmented = [label for label in y]

        for _ in range(num_copies):
            for i in range(len(X)):
                x = X[i]
                noise = np.random.normal(0, feature_std * noise_factor)
                x_new = x + noise
                X_augmented.append(x_new)
                y_augmented.append(y[i])

        return np.array(X_augmented), np.array(y_augmented)

    def select_features(self, k=30):
        '''
        Selects the best k features using SelectKBest.
        '''

        self.selector = SelectKBest(k=k)
        self.X = self.selector.fit_transform(self.X, self.y)

    def train_model(self, max_depth=10, n_estimators=50, test_size=0.2, use_augmentation=True):
        '''
        Trains a Random Forest model on the data with optional augmentation.
        '''

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size)

        if use_augmentation:
            X_train, y_train = self.augment_with_noise(
                X_train, y_train, noise_factor=0.1, num_copies=3)

        self.classifier = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators, n_jobs=-1, verbose=1)

        self.classifier.fit(X_train, y_train)

        # Calculating and printing ROC AUC score
        print(
            f'\n\nROC AUC on 20% training data: {roc_auc_score(y_test, [i[1] for i in self.classifier.predict_proba(X_test)])}\n\n')

    def retrain_on_full_dataset(self, use_augmentation=True):
        '''
        Retrains the model on the full dataset for final predictions.
        '''

        if use_augmentation:
            X_aug, y_aug = self.augment_with_noise(
                self.X, self.y, noise_factor=0.1, num_copies=3)

            self.classifier.fit(X_aug, y_aug)

        else:
            self.classifier.fit(self.X, self.y)

    def predict(self):
        '''
        Generates predictions on test data and saves to output file.
        '''

        # Transform test data using the same feature selector
        X_test = self.selector.transform(self.test_data.iloc[:, 1:].values)

        # Generate probability predictions
        y_pred = [i[1] for i in self.classifier.predict_proba(X_test)]

        # Save predictions to CSV
        with open(self.output_path, 'w') as f:
            f.write("ID,Labels\n")

            for i, prob in enumerate(y_pred, 1001):
                f.write(f"{i},{prob}\n")

        print(f"\n\nPredictions saved to {self.output_path}\n\n")


def main():
    '''
    Parses command-line arguments and runs the cancer classification workflow.
    '''

    parser = argparse.ArgumentParser(
        description='Cancer Classification using Gene Expression Data')
    parser.add_argument('--train', required=True,
                        help='Path to training CSV file')
    parser.add_argument('--test', required=True, help='Path to test CSV file')
    parser.add_argument('--output', required=True,
                        help='Path to save predictions CSV file')
    parser.add_argument('--features', type=int, default=30,
                        help='Number of features to select')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum depth of Random Forest')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='Number of estimators in Random Forest')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')

    args = parser.parse_args()

    classifier = CancerClassifier(args.train, args.test, args.output)

    classifier.select_features(k=args.features)

    classifier.train_model(max_depth=args.max_depth,
                           n_estimators=args.n_estimators,
                           use_augmentation=not args.no_augmentation)

    classifier.retrain_on_full_dataset(
        use_augmentation=not args.no_augmentation)

    classifier.predict()


if __name__ == "__main__":

    main()
