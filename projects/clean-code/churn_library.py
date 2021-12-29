"""
Source code for complete analysis and training of the Churn model
"""

import os
import argparse
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve

from constants import CAT_COLUMNS, QUANT_COLUMNS, ATTRITION_COLUMN, \
    FIGSIZE, PARAM_GRID

# Logger
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler('churn_log.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()


class ChurnModel:
    """
    End-to-end process to create a ML model for Churn.
    - Read input data
    - Perform EDA
    - Preprocess data
    - Develop models
    - Output performances
    """

    def __init__(self,
                 category_columns: list = CAT_COLUMNS,
                 numerical_columns: list = QUANT_COLUMNS,
                 attrition_flag: str = ATTRITION_COLUMN):

        self.category_columns = category_columns
        self.numerical_columns = numerical_columns
        self.attrition_flag = attrition_flag
        self.encodings = {}
        self.rfc = RandomForestClassifier(random_state=42)
        self.lrc = LogisticRegression(solver='liblinear')
        self.input_data = None

    def import_data(self, pth: str):
        """
        Return dataframe for the csv found at pth

        Parameters
        ----------
        pth: str
            Path to the csv

        Returns
        -------
        df: pd.DataFrame
        """
        # Check if pth is a string
        try:
            assert isinstance(pth, str)
        except AssertionError:
            logger.error(f'{pth} should be a string')
        # Check if pth file exists
        try:
            assert os.path.isfile(pth)
        except AssertionError:
            logger.error(f'File {pth} does not exist')
        logger.info(f'Read data from {pth}')
        self.input_data = pd.read_csv(pth)
        logger.info(f'Input data has {self.input_data.shape[0]} records')

    def perform_eda(self,
                    figsize: tuple = FIGSIZE,
                    kde: bool = False):
        """
        Perform eda on df and save figures to images folder

        Parameters
        ----------
        figsize: tuple
            Figsize for all plots
        kde: bool, default False
            If True, apply kde to all numerical variables' plots
        """

        try:
            assert len(figsize) == 2
        except AssertionError:
            logger.error(f'{figsize} should be a tuple of 2 elements')
        try:
            assert isinstance(kde, bool)
        except AssertionError:
            logger.error(f'{kde} should be a boolean')

        # Create comprehensive list with all features
        all_features = self.category_columns + self.numerical_columns + \
                       [self.attrition_flag, 'correlations']

        logging.info(f'Use {figsize} as figsize for all plots')
        logging.info(f"Use kde={kde} for all numerical variables' plots")
        # Cycle through the features and create the right plot type
        for feat in all_features:
            plt.figure(figsize=figsize)
            if feat == self.attrition_flag:
                logging.info(f'Histogram on target variable {feat}')
                self.input_data[feat].hist()
            elif feat == 'correlations':
                logging.info('Heatmap with correlation coefficients')
                sns.heatmap(self.input_data[self.numerical_columns].corr(),
                            annot=False, cmap='Dark2_r', linewidths=2)
            elif feat in self.category_columns:
                logging.info(f'Barchart on categorical variable {feat} with '
                             'relative frequency')
                self.input_data[feat].value_counts('normalize').plot(kind='bar')
            else:
                logging.info(f'Histogram on numerical variable {feat}')
                sns.histplot(self.input_data[feat], kde=kde)
            logging.info(f'Save plot to images/{feat}.png')
            plt.savefig(f'images/{feat}.png')

    def target_encoder_fit(self, input_data, label_data):
        """
        Fit target encoder based on train data

        Parameters
        ----------
        input_data: pd.DataFrame
            Input data used to calculate encodings
        label_data: pd.Series
            Target data
        """

        len_category_lst = len(self.category_columns)
        for i, feature in enumerate(self.category_columns):
            j = i + 1
            logging.info(f'Encode categorical feature {feature} - {j} of '
                         f'{len_category_lst}')
            # Calculate encodings on input data
            self.encodings[feature] = pd.concat(
                (input_data[feature], label_data), 1
            ).groupby(feature)['Churn'].mean()

    def target_encoder_apply(self, input_data):
        """
        Apply fitted target encorder to input data

        Parameters
        ----------
        input_data: pd.DataFrame
            Input data where to apply encodings

        Returns
        -------
        x_encoded: pd.DataFrame
            Dataframe with encoded columns
        """

        x_encoded = input_data.copy()
        len_category_lst = len(self.encodings)
        if len_category_lst == 0:
            logging.error('Target encoding is still not fitted')
            raise AttributeError('Target encoding is still not fitted')
        for i, feature in enumerate(self.encodings):
            j = i + 1
            logging.info(f'Encode categorical feature {feature} - {j} of '
                         f'{len_category_lst}')
            # Apply encodings on input data
            x_encoded[feature] = pd.merge(
                input_data[feature], self.encodings[feature],
                left_on=feature, right_index=True
            )['Churn']

        return x_encoded

    def perform_feature_engineering(self,
                                    test_size: float = 0.3):
        """
        Perform feature engineering:
        - create binary target
        - split train/test
        - encode categorical features

        Parameters
        ----------
        test_size: float, default 0.3
            Percentage for test data

        Returns
        -------
        x_train: pd.DataFrame
            X training data
        x_test: pd.DataFrame
            X testing data
        y_train: pd.Series
            y training data
        y_test: pd.Series
            y testing data
        """

        if self.input_data is None:
            logging.error('Input data is missing')
            raise AttributeError('Input data is missing')
        # Encode response variable
        logging.info(f'Encode {self.attrition_flag} feature into binary '
                     'feature Churn')
        self.input_data['Churn'] = 1
        self.input_data.loc[
            self.input_data[self.attrition_flag] == 'Existing Customer',
            'Churn'
        ] = 0

        # Select only relevant features
        self.input_data = self.input_data[
            self.category_columns + self.numerical_columns + ['Churn']
        ]

        # Split train-test
        logging.info(f'Split input data into train/test ({test_size} test '
                     'size)')
        x_train, x_test, y_train, y_test = train_test_split(
            self.input_data.drop('Churn', 1), self.input_data['Churn'],
            test_size=test_size, random_state=42,
            stratify=self.input_data['Churn']
        )

        logging.info('Encode categorical data')
        self.target_encoder_fit(x_train, y_train)
        x_train = self.target_encoder_apply(x_train)
        x_test = self.target_encoder_apply(x_test)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def _predict_and_store(estimator, input_data, pth):

        preds = pd.Series(estimator.predict(input_data))
        preds.to_csv(pth, index=False)
        return preds

    def train_models(self,
                     param_grid: dict = PARAM_GRID):
        """
        train, store model results: images + scores, and store models
        """

        if self.input_data is None:
            logging.error('Input data is missing')
            raise AttributeError('Input data is missing')

        x_train, x_test, y_train, y_test = self.perform_feature_engineering()

        cv_rfc = GridSearchCV(estimator=self.rfc,
                              param_grid=param_grid,
                              cv=5,
                              verbose=3)
        cv_rfc.fit(x_train, y_train)

        self.lrc.fit(x_train, y_train)

        self.rfc = cv_rfc.best_estimator_

        # Save models
        logging.info('Save random forest model to models/rfc.pkl')
        joblib.dump(self.rfc, 'models/rfc.pkl')
        logging.info('Save logistic regression model to models/lrc.pkl')
        joblib.dump(self.lrc, 'models/lrc.pkl')

        # Get predictions
        logging.info('Calculate predictions for random forest')
        y_train_preds_rf = self._predict_and_store(self.rfc,
                                                   x_train,
                                                   'data/rfc_preds_train.csv')
        y_test_preds_rf = self._predict_and_store(self.rfc,
                                                  x_test,
                                                  'data/rfc_preds_test.csv')

        logging.info('Calculate predictions for logistic regression')
        y_train_preds_lr = self._predict_and_store(self.lrc, x_train,
                                                   'data/lrc_preds_train.csv')
        y_test_preds_lr = self._predict_and_store(self.lrc, x_test,
                                                  'data/lrc_preds_test.csv')

        self.classification_report_image(y_train, x_test, y_test,
                                         y_train_preds_lr, y_train_preds_rf,
                                         y_test_preds_lr, y_test_preds_rf)

        logging.info('Get feature importance for random forest')
        self.feature_importance_plot(self.rfc, x_test,
                                     'images/rfc_feature_importances.png')

    def classification_report_image(self,
                                    y_train,
                                    x_test,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf,
                                    figsize: tuple = FIGSIZE):
        """
        Produce classification report for training and testing results, and
        stores report as image in images folder

        Parameters
        ----------
        y_train: pd.Series
            Training response values
        x_test: pd.DataFrame
            Test input data
        y_test: pd.Series
            Test response values
        y_train_preds_lr: pd.Series
            Training predictions from logistic regression
        y_train_preds_rf: pd.Series
            Training predictions from random forest
        y_test_preds_lr: pd.Series
            Test predictions from logistic regression
        y_test_preds_rf: pd.Series
            Test predictions from random forest
        figsize: tuple
            Figsize for all plots
        """
        logging.info('Calculate random forest results')
        print('random forest results')
        print('test results')
        print(classification_report(y_test, y_test_preds_rf))
        print('train results')
        print(classification_report(y_train, y_train_preds_rf))

        logging.info('Calculate logistic regression results')
        print('logistic regression results')
        print('test results')
        print(classification_report(y_test, y_test_preds_lr))
        print('train results')
        print(classification_report(y_train, y_train_preds_lr))

        # Create ROC Curve image
        logging.info('Plot ROC Curve for both models')
        plt.figure(figsize=figsize)
        figure_ax = plt.gca()
        plot_roc_curve(self.lrc, x_test, y_test, ax=figure_ax)
        plot_roc_curve(self.rfc, x_test, y_test, ax=figure_ax, alpha=0.8)
        logging.info('Save plot to images/ROC.png')
        plt.savefig('images/ROC.png')

    @staticmethod
    def feature_importance_plot(estimator,
                                input_data: pd.DataFrame,
                                output_pth: str,
                                figsize: tuple = FIGSIZE):
        """
        creates and stores the feature importances in pth

        Parameters
        ----------
            estimator: scikit-learn estimator
                Model object containing feature_importances_
            input_data: pd.DataFrame
                Input data
            output_pth: str
                Path where to store the figure
        figsize: tuple
            Figsize for all plots
        """

        plt.figure(figsize=figsize)

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(input_data)
        shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)

        logging.info(f'Save feature importances plot to {output_pth}')
        plt.savefig(output_pth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--input-path',
                        help='Path to input data',
                        default='./data/bank_data.csv')
    args = parser.parse_args()

    model = ChurnModel()
    model.import_data(args.input_path)
    model.perform_eda()
    model.train_models()
