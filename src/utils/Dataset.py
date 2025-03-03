import openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Dataset:
    @staticmethod
    def load_data(dataset_id):
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        X = X.dropna()
        y = y.loc[X.index]

        assert len(np.unique(y)) == 2

        if y.dtype == 'object' or y.dtype.name == 'category':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        return X.to_numpy(), y
