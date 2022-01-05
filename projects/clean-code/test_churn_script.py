"""
Test suite for churn_library.py code - use with pytest
"""

import os
import logging

import pandas as pd
import numpy as np

from churn_library import ChurnModel

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    test data import - this example is completed for you to assist with the
    other test functions
    """
    model = ChurnModel()
    model.import_data("./data/bank_data.csv")
    assert model.input_data.shape[0] > 0 and model.input_data.shape[1] > 0


def test_eda():
    """
    test perform eda function
    """

    model = ChurnModel()
    model.import_data("./data/bank_data.csv")
    model.perform_eda()
    for feat in (
        model.category_columns
        + model.numerical_columns
        + [model.attrition_flag, "correlations"]
    ):
        assert os.path.isfile(f"images/{feat}.png")


class TestFeatureEngineering:
    """
    group all tests on feature engineering
    """

    def initialize(self):
        model = ChurnModel()
        model.import_data("./data/bank_data.csv")
        x_train, x_test, y_train, y_test = model.perform_feature_engineering()
        return x_train, x_test, y_train, y_test

    def test_feat_eng_types(self):
        """
        test output types
        """
        x_train, x_test, y_train, y_test = self.initialize()
        assert (
            isinstance(x_train, pd.DataFrame)
            and isinstance(x_test, pd.DataFrame)
            and isinstance(y_train, pd.Series)
            and isinstance(y_test, pd.Series)
        )

    def test_feat_eng_data(self):
        """
        test data is present
        """
        x_train, x_test, y_train, y_test = self.initialize()
        assert (
            (x_train.shape[0] > 0 and x_train.shape[1] > 0)
            and (x_test.shape[0] > 0 and x_test.shape[1] > 0)
            and y_train.shape[0] > 0
            and y_test.shape[0] > 0
        )

    def test_feat_eng_stratified_split(self):
        """
        test output has equal mean
        """
        x_train, x_test, y_train, y_test = self.initialize()
        assert np.abs(y_train.mean() - y_test.mean()) < 1e-3

    def test_feat_eng_all_numerical(self):
        """
        test all features are numerical
        """
        x_train, x_test, y_train, y_test = self.initialize()
        assert (
            len([x for x in x_train.dtypes if x not in ("float", "int")]) == 0
            and len([x for x in x_test.dtypes if x not in ("float", "int")]) == 0
        )


class TestTrainModels:
    """
    group all tests on training models
    """

    def initialize(self):
        model = ChurnModel()
        model.import_data("./data/bank_data.csv")
        model.train_models(param_grid={"n_estimators": [50]})

    def test_train_models_export_pkl(self):
        """
        test files with exported models exist
        """
        self.initialize()
        assert os.path.isfile("models/rfc.pkl") and os.path.isfile("models/lrc.pkl")

    def test_train_models_export_predictions(self):
        """
        test files with predictions exist
        """
        self.initialize()
        assert (
            os.path.isfile("data/rfc_preds_train.csv")
            and os.path.isfile("data/rfc_preds_test.csv")
            and os.path.isfile("data/lrc_preds_train.csv")
            and os.path.isfile("data/lrc_preds_test.csv")
        )

    def test_train_models_export_feature_importance(self):
        """
        test feature importance plot exists
        """
        self.initialize()
        assert os.path.isfile("images/rfc_feature_importances.png")
