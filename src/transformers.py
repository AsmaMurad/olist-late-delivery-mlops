import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = [str(column) for column in X.columns]
        self.frequency_maps_ = {}

        for column in X.columns:
            values = X[column].copy()
            values = values.astype(object)
            values = values.where(values.notna(), "__MISSING__")
            self.frequency_maps_[column] = values.value_counts().to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        output = pd.DataFrame(index=X.index)

        for column in X.columns:
            values = X[column].copy()
            values = values.astype(object)
            values = values.where(values.notna(), "__MISSING__")
            output[f"{column}_frequency"] = values.map(
                self.frequency_maps_[column]
            ).fillna(0)

        return output

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.array(
            [f"{feature}_frequency" for feature in input_features],
            dtype=object,
        )
