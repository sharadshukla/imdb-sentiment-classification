"""
Classical machine-learning model configurations for IMDb sentiment classification.

Each function returns a GridSearchCV object containing:
- binary CountVectorizer feature extraction
- the classifier
- the hyperparameter grid used in the original experiment
- 5-fold cross-validation with F1 scoring

Models included:
- Logistic Regression
- Bernoulli Naive Bayes
- LinearSVC
- Random Forest
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def create_logistic_regression_search():
    """
    Create the Logistic Regression GridSearchCV configuration.
    """

    pipeline = Pipeline([
        (
            "vec",
            CountVectorizer(
                binary=True,
                min_df=5,
                stop_words="english"
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    param_grid = {
        "vec__max_features": [20000, 50000],
        "clf__C": [0.01, 0.1, 1, 10]
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )


def create_naive_bayes_search():
    """
    Create the Bernoulli Naive Bayes GridSearchCV configuration.
    """

    pipeline = Pipeline([
        (
            "vec",
            CountVectorizer(
                binary=True,
                min_df=5,
                stop_words="english"
            )
        ),
        (
            "clf",
            BernoulliNB()
        )
    ])

    param_grid = {
        "vec__max_features": [20000, 50000],
        "clf__alpha": [0.01, 0.1, 0.5, 1.0]
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )


def create_linearsvc_search():
    """
    Create the LinearSVC GridSearchCV configuration.
    """

    pipeline = Pipeline([
        (
            "vec",
            CountVectorizer(
                binary=True,
                min_df=5,
                stop_words="english"
            )
        ),
        (
            "clf",
            LinearSVC(
                max_iter=2000,
                random_state=42
            )
        )
    ])

    param_grid = {
        "vec__max_features": [20000, 50000],
        "clf__C": [0.01, 0.1, 1, 10]
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )


def create_random_forest_search():
    """
    Create the Random Forest GridSearchCV configuration.
    """

    pipeline = Pipeline([
        (
            "vec",
            CountVectorizer(
                binary=True,
                min_df=5,
                stop_words="english"
            )
        ),
        (
            "clf",
            RandomForestClassifier(
                random_state=42
            )
        )
    ])

    param_grid = {
        "vec__max_features": [20000, 50000],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 20, 50],
        "clf__min_samples_leaf": [1, 2]
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1
    )