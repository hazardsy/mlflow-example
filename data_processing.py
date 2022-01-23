from typing import List, Tuple
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(
    train_test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List]:
    """Get covtype data, split & scaled.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List]: Training X, Testing X, Training Y, Testing Y, Target names
    """
    dataset = fetch_covtype()

    feature_names = list(dataset.feature_names)
    df = pd.DataFrame(dataset.data, columns=feature_names)

    X = df
    y = pd.Series(dataset.target)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=train_test_ratio, random_state=42
    )

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test, dataset.target_names
