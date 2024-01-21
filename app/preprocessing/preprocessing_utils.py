import logging

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def VIF_feature_selection(
    data: pd.DataFrame, threshold: float, exclude_columns: list
) -> pd.Index:
    logging.info("Start vif procedure")
    X = data.drop(columns=exclude_columns, errors="ignore")

    vif = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    inf_columns_index = np.array(vif) == np.inf
    logging.info(f"Remove columns with inf VIF: {X.columns[inf_columns_index]}")
    X = X.drop(columns=X.columns[inf_columns_index])

    vif = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    while np.any(np.array(vif) > threshold):
        logging.info(f"Remove {X.columns[np.argmax(vif)]} column")
        X = X.drop(columns=X.columns[np.argmax(vif)])
        vif = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    return exclude_columns + list(X.columns)
