"""Count Encoding"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.utils import get_obj_cols, convert_input
from sklearn.utils.random import check_random_state


class CountEncoder(BaseEstimator, TransformerMixin):
    """Count encoder for categorical features. Returns the frequency of every category.

    Parameters
    ----------
    cols: list
        a list of columns to encode, if None, all string columns will be encoded
    drop_invariant: bool
        boolean for whether or not to drop columns with 0 variance
    return_df: bool
        boolean for whether to return a pandas DataFrame from transform (otherwise it will be a numpy array)
    randomized : boolean, Add normal (Gaussian) distribution randomized to the encoder or not
    sigma : float, Standard deviation (spread or "width") of the distribution.

    """

    def __init__(self, cols=None, drop_invariant=False, return_df=True, random_state=None, randomized=False,
                 sigma=0.05):
        self.return_df = return_df
        self.drop_invariant = drop_invariant
        self.drop_cols = []
        self.cols = cols
        self._dim = None
        self.mapping = None
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma

    def fit(self, X, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------

        self : encoder
            Returns self.

        """

        # first check the type
        X = convert_input(X)

        self._dim = X.shape[1]

        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)
        self.random_state_ = check_random_state(self.random_state)

        X_temp, categories = self.count_encoder(X, mapping=self.mapping, cols=self.cols)

        self.mapping = categories

        if self.drop_invariant:
            self.drop_cols = [x for x in self.cols if X_temp[x].var() <= 10e-5]
        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]

        Returns
        -------

        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.

        """

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
        X = convert_input(X)

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        if not self.cols:
            return X
        X, _ = self.count_encoder(X, mapping=self.mapping, cols=self.cols)

        if self.drop_invariant:
            for col in self.drop_cols:
                X.drop(col, 1, inplace=True)

        if self.return_df:
            return X
        else:
            return X.values

    def count_encoder(self, X_in, mapping=None, cols=None):
        """
        Count encoding uses a single column of float to represent the frequency of a category
        :param X_in: array like dataset that will be encoded
        :param mapping: a list of length = len(cols) containing dictionaries
        Each dictionary is of the type {'col': 'Gender', 'mapping': {'female': 2, 'male': 1}}
        where col is the name of the column being encoded (e.g. Gender),
        and mapping is a dict with keys = categories of the categorical features and value = their count
        :param cols: the columns to be encoded

        """

        X = X_in.copy(deep=True)

        if cols is None:
            cols = X.columns.values

        if mapping is not None:
            mapping_out = mapping
            for switch in mapping:
                X[str(switch.get('col')) + '_tmp'] = np.nan

                # loop through the categories of the feature e.g.Female, Male
                for val in switch.get('mapping'):
                    X.loc[X[switch.get('col')] == val, str(switch.get('col')) + '_tmp'] = \
                        switch.get('mapping')[val]
                del X[switch.get('col')]
                X.rename(columns={str(switch.get('col')) + '_tmp': switch.get('col')}, inplace=True)

                if self.randomized:
                    X[switch.get('col')] = (X[switch.get('col')] *
                                            self.random_state_.normal(1., self.sigma, X[switch.get('col')].shape[0]))

                X[switch.get('col')] = X[switch.get('col')].astype(float).values.reshape(-1, )
        else:
            mapping_out = []

            for col in cols:
                tmp = X[col].value_counts()
                tmp = tmp.to_dict()

                X[str(col) + '_tmp'] = np.nan
                for val in tmp:
                    X.loc[X[col] == val, str(col) + '_tmp'] = tmp[val]
                del X[col]
                X.rename(columns={str(col) + '_tmp': col}, inplace=True)

                X[col] = X[col].astype(float).values.reshape(-1, )

                mapping_out.append({'col': col, 'mapping': tmp}, )

        return X, mapping_out


if __name__ == '__main__':
    X = pd.DataFrame(
        [['female', 'New York', 'low', 4], ['female', 'London', 'medium', 3], ['male', 'New Delhi', 'high', 2]],
        columns=['Gender', 'City', 'Temperature', 'Rating'])
    enc = CountEncoder(cols=['Gender', 'City'], drop_invariant=True, randomized=True).fit(X)
    numeric_dataset = enc.transform(X)
    print(numeric_dataset)
