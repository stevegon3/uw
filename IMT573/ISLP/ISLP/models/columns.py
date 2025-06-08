
from typing import NamedTuple, Any
from copy import copy

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder)
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class Column(NamedTuple):

    """
    A column extractor with a possible
    encoder (following `sklearn` fit/transform template).
    """

    idx: Any
    name: str
    is_categorical: bool = False
    is_ordinal: bool = False
    columns: tuple = ()
    encoder: Any = None

    def get_columns(self, X, fit=False):

        """
        Extract associated column from X,
        encoding it with `self.encoder` if not None.

        Parameters
        ----------

        X : array-like
            X on which model matrix will be evaluated.
            If a `pd.DataFrame` or `pd.Series`, variables that are of
            categorical dtype will be treated as categorical.

        fit : bool
            If True, fit `self.encoder` on corresponding
            column.

        Returns
        -------
        cols : `np.ndarray`
            Evaluated columns -- if an encoder is used,
            several columns may be produced.

        names : (str,)
            Column names
        """

        cols = _get_column(self.idx, X) 
        if fit:
            self.fit_encoder(X)

        if self.encoder is not None:
            try:
                cols = self.encoder.transform(cols)
            except ValueError:
                # commonly raised for transformers given pd.Seriies
                # try casting to a 2-d ndarray
                cols = self.encoder.transform(np.asarray(cols).reshape((-1,1)))
        cols = np.asarray(cols)

        names = self.columns
        if hasattr(self.encoder, 'columns_'):
            names = ['{0}[{1}]'.format(self.name, c)
                     for c in self.encoder.columns_]
        if not names:
            names = ['{0}[{1}]'.format(self.name, i)
                     for i in range(cols.shape[1])]
        return cols, names

    def fit_encoder(self, X):

        """
        Fit `self.encoder`.

        Parameters
        ----------
        X : array-like
            X on which encoder will be fit.

        Returns
        -------
        None
        """
        cols = _get_column(self.idx, X)
        if self.encoder is not None:
            try:
                check_is_fitted(self.encoder)
            except NotFittedError:
                self.encoder.fit(cols)
        return np.asarray(cols)


# private functions


def _get_column(idx,
                X,
                loc=True):
    """
    Extract column `idx` from `X`
    as a two-dimensional ndarray or a pd.DataFrame
    """
    if isinstance(X, np.ndarray):
        col = X[:, [idx]]
    elif hasattr(X, 'loc'):
        if loc:
            col = X.loc[:, [idx]]
        else:   # use iloc instead
            col = X.iloc[:, [idx]]
    else:
        raise ValueError('expecting an ndarray or a ' +
                         '"loc/iloc" methods, got %s' % str(X))

    return col

    
def _get_column_info(X,
                     columns,
                     is_categorical,
                     is_ordinal,
                     categorical_encoders={}
                     ):


    """
    Compute a dictionary
    of `Column` instances for each column
    of `X`. Keys are `columns`.

    Categorical and ordinal columns use the
    default encoding provided.

    """

    column_info = {}
    for i, col in enumerate(columns):
        if type(col) == int:
            name = f'X{col}'
        else:
            name = str(col)
        if is_categorical[i]:
            if is_ordinal[i]:
                Xcol = _get_column(col, X) 
                if col not in categorical_encoders:
                    encoder = clone(categorical_encoders['ordinal'])
                else:
                    encoder = categorical_encoders[col]
                encoder.fit(Xcol)
                columns = ['{0}'.format(col)]
            else:
                Xcol = _get_column(col, X) 
                if col not in categorical_encoders:
                    encoder = clone(categorical_encoders['categorical'])
                else:
                    encoder = categorical_encoders[col]
                cols = encoder.fit_transform(Xcol)
                if hasattr(encoder, 'columns_'):
                    columns_ = encoder.columns_
                else:
                    columns_ = range(cols.shape[1])
                columns = ['{0}[{1}]'.format(col, c)
                           for c in columns_]
            column_info[col] = Column(col,
                                      name,
                                      is_categorical[i],
                                      is_ordinal[i],
                                      tuple(columns),
                                      encoder)
        else:
            Xcol = _get_column(col, X) 
            column_info[col] = Column(col,
                                      name,
                                      columns=(name,))
    return column_info

# extracted from method of BaseHistGradientBoosting from
# https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
# max_bins is ignored

def _check_categories(categorical_features, X):
    """Check and validate categorical features in X

    Return
    ------
    is_categorical : ndarray of shape (n_features,) or None, dtype=bool
        Indicates whether a feature is categorical. If no feature is
        categorical, this is None.
    known_categories : list of size n_features or None
        The list contains, for each feature:
            - an array of shape (n_categories,) with the unique cat values
            - None if the feature is not categorical
        None if no feature is categorical.
    """
    if categorical_features is None:
        return None, None

    categorical_features = np.asarray(categorical_features)

    if categorical_features.size == 0:
        return None, None

    if categorical_features.dtype.kind not in ('i', 'b'):
        raise ValueError("categorical_features must be an array-like of "
                         "bools or array-like of ints.")

    n_features = X.shape[1]

    # check for categorical features as indices
    if categorical_features.dtype.kind == 'i':
        if (np.max(categorical_features) >= n_features
                or np.min(categorical_features) < 0):
            raise ValueError("categorical_features set as integer "
                             "indices must be in [0, n_features - 1]")
        is_categorical = np.zeros(n_features, dtype=bool)
        is_categorical[categorical_features] = True
    else:
        if categorical_features.shape[0] != n_features:
            raise ValueError("categorical_features set as a boolean mask "
                             "must have shape (n_features,), got: "
                             f"{categorical_features.shape}")
        is_categorical = categorical_features

    if not np.any(is_categorical):
        return None, None

    # compute the known categories in the training data. We need to do
    # that here instead of in the BinMapper because in case of early
    # stopping, the mapper only gets a fraction of the training data.
    known_categories = []

    for f_idx in range(n_features):
        if is_categorical[f_idx]:
            categories = np.array([v for v in
                                   set(_get_column(f_idx,
                                                   X,
                                                   loc=False))])
            missing = []
            for c in categories:
                try:
                    missing.append(np.isnan(c))
                except TypeError:  # not a float
                    missing.append(False)
            missing = np.array(missing)
            if missing.any():
                categories = sorted(categories[~missing])

        else:
            categories = None
        known_categories.append(categories)

    return is_categorical, known_categories


def _categorical_from_df(df):
    """
    Find
    """
    is_categorical = []
    is_ordinal = []
    for c in df.columns:
        try:
            if df[c].dtype == 'category':
                is_categorical.append(True)
                is_ordinal.append(df[c].cat.ordered)
            else:
                is_categorical.append(False)
                is_ordinal.append(False)
        except (TypeError, AttributeError):
            is_categorical.append(False)
            is_ordinal.append(False)
    is_categorical = np.asarray(is_categorical)
    is_ordinal = np.asarray(is_ordinal)

    return is_categorical, is_ordinal
