from functools import partial
from os.path import join, abspath, dirname

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


DATA_ROOT_PATH = join(dirname(abspath(__file__)), "..", "data")

class DatasetLoader:
    """Utility to load, format and split (train/test) datasets"""

    def __init__(self, data_path=DATA_ROOT_PATH, test_size=0.25, random_state=42):
        self.data_path = data_path
        self.random_state = check_random_state(random_state)
        self.test_size = test_size

    def load(self, dataset):
        dataset_loaders = {'ads': self._load_ads,
                           'adult': self._load_adult,
                           'breast': self._load_breast,
                           'mnist17': partial(self._load_mnist, low=1, high=7),
                           'mnist49': partial(self._load_mnist, low=4, high=9),
                           'mnist56': partial(self._load_mnist, low=5, high=6),
                           'mnist_low_high': partial(self._load_full_mnist, task='low_vs_high'),
                           'mnist_even_odd': partial(self._load_full_mnist, task='even_vs_odd')}

        if dataset not in dataset_loaders.keys():
            raise RuntimeError(f"Invalid dataset {dataset}")

        return dataset_loaders[dataset]()

    def _load_breast(self):
        breast = load_breast_cancer()
        X = StandardScaler().fit_transform(breast.data)
        y = (2 * breast.target - 1).reshape(-1, 1)
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)


    def _load_mnist(self, low, high):
        X_low = np.loadtxt(join(self.data_path, "mnist", f"mnist_{low}")) / 255
        y_low = -1 * np.ones(X_low.shape[0])

        X_high = np.loadtxt(join(self.data_path, "mnist", f"mnist_{high}")) / 255
        y_high = np.ones(X_high.shape[0])

        X = np.vstack((X_low, X_high))
        y = np.hstack((y_low, y_high)).reshape(-1, 1)

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def _load_full_mnist(self, task='low_vs_high'):
        if task == 'low_vs_high':
            X_low = np.vstack([np.loadtxt(join(self.data_path, "mnist", f"mnist_{n}")) \
                                                                            for n in [0, 1, 2, 3, 4]]) / 255
            y_low = -1 * np.ones(X_low.shape[0])

            X_high = np.vstack([np.loadtxt(join(self.data_path, "mnist", f"mnist_{n}")) \
                                                                            for n in [5, 6, 7, 8, 9]]) / 255
            y_high = np.ones(X_high.shape[0])

            X = np.vstack((X_low, X_high))
            y = np.hstack((y_low, y_high)).reshape(-1, 1)

        elif task == 'even_vs_odd':
            X_even = np.vstack([np.loadtxt(join(self.data_path, "mnist", f"mnist_{n}")) \
                                                                            for n in [0, 2, 4, 6, 8]]) / 255
            y_even = -1 * np.ones(X_even.shape[0])

            X_odd = np.vstack([np.loadtxt(join(self.data_path, "mnist", f"mnist_{n}")) \
                                                                            for n in [1, 3, 5, 7, 9]]) / 255
            y_odd = np.ones(X_odd.shape[0])

            X = np.vstack((X_even, X_odd))
            y = np.hstack((y_even, y_odd)).reshape(-1, 1)

        else:
            raise RuntimeError(f"Invalid MNIST binary task: {task}")

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def _load_ads(self):
        df = pd.read_csv(join(self.data_path, "ad.data"), sep=",", header=None)

        le = LabelEncoder()
        y = df.iloc[:, -1]
        y = (2 * le.fit_transform(y) - 1).reshape(-1, 1)

        # We use all but the first 4 features which are sometimes missing in the data.
        X = StandardScaler().fit_transform(df.iloc[:, 4:-1].values)

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def _load_adult(self):
        features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        categorical_features = [f for i, f in enumerate(features) if i in [1, 3, 5, 6, 7, 8, 9, 13]]

        df_train = pd.read_csv(join(self.data_path, "adult.data"), sep=",", header=None)
        df_test = pd.read_csv(join(self.data_path, "adult.test"), sep=",", header=None)

        df = df_train.append(df_test)

        le = LabelEncoder()
        y = df.iloc[:, -1]
        y = y.str.strip(to_strip='.')
        y = (2 * le.fit_transform(y) - 1).reshape(-1, 1)

        X = df.iloc[:, :-1]
        X = X.rename(columns={i:f for i, f in enumerate(features)})
        X = pd.get_dummies(X, columns=categorical_features)
        X = X.drop([c for c in X.columns.values if '_?' in c], axis=1)

        X = StandardScaler().fit_transform(X)

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
