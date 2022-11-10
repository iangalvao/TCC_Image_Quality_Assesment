# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import sys
from tabnanny import verbose
import pandas as pd
import numpy as np
import yaml
from itertools import chain, combinations
import datetime
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.utils import shuffle

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID


sklearn_models = {
    "LogisticRegression": LogisticRegression,
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
}


def load_config(config_file_path):
    with open("src/" + config_file_path, "r") as file:
        config = yaml.full_load(file)
        print(config)
    return config


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def eval_metrics_cv(cv):
    print(cv)
    print(cv.keys())
    rmse = cv["test_neg_root_mean_squared_error"].mean()
    mae = cv["test_neg_mean_absolute_error"].mean()
    r2 = cv["test_r2"].mean()
    return rmse, mae, r2


def selectFeatures(col_names, df):
    X = df[col_names]
    return X


def selectTarget(col_name, df):
    y = df[col_name]
    return y


def powerset(l, start=1, end=0):
    assert start > 0
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    end = min(end + 1, len(l) + 1)
    return chain.from_iterable(combinations(l, r) for r in range(start, end))


def cross_validation(model, _X, _y, _cv):
    _scoring = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
    results = cross_validate(estimator=model, X=_X, y=_y, cv=_cv, scoring=_scoring)
    return results


def mlflow_log(params=None, tags=None, metrics=None, model=None):
    for key, value in params.items():
        mlflow.log_param(key, value)
    for key, value in tags.items():
        mlflow.set_tag(key, value)
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    if model:
        mlflow.log_model(model, "model")


def training(model, X, y):
    model.fit(X, y)
    return model


def metrics_from_predict(model, X, y):
    print("Training Complete. Predicting")
    predicted_qualities = model.predict(X)
    (rmse, mae, r2) = eval_metrics(y, predicted_qualities)
    return rmse, mae, r2


def load_grid_params(config, model_name):
    grid_params = config[model_name]["GridSearch"]
    param_grid = {}
    if "Range" in grid_params.keys():
        range_params = grid_params["Range"]
        for k, v in range_params.items():
            param_grid[k] = range(v[0], v[1], v[2])
    if "List" in grid_params.keys():
        list_params = grid_params["List"]
        for k, v in list_params.items():
            param_grid[k] = v

    return param_grid


def create_child_run(parent_run_id, experiment_id):
    child_run_1 = client.create_run(
        experiment_id=experiment_id, tags={MLFLOW_PARENT_RUN_ID: parent_run_id}
    )

    client.log_param(child_run_1.info.run_id, "who", "child")
    return child_run_1


def log_run(
    gridsearch: GridSearchCV,
    model_name: str,
    run_index: int,
    tags,
    experiment_id,
    run_id,
):
    """Logging of cross validation results to mlflow tracking server

    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        run_index (int): Index of the run (in Gridsearch)
        tags (dict): Dictionary of extra data and tags (usually features)
    """

    cv_results = gridsearch.cv_results_
    print("run/experimendt id:", run_id, experiment_id)
    child_run = create_child_run(run_id, experiment_id)
    print("CREATE RUN OK")
    child_id = child_run.info.run_id
    client.log_param(child_id, "folds", gridsearch.cv)

    print("Logging parameters")
    params = list(gridsearch.param_grid.keys())
    for param in params:
        client.log_param(child_id, param, cv_results["param_%s" % param][run_index])

    print("Logging metrics. Printing CV results")
    # print(cv_results)
    for score_name in [score for score in cv_results if "mean_test" in score]:
        if ("absolute" in score_name) or ("squared" in score_name):
            client.log_metric(
                child_id, score_name, (-1) * cv_results[score_name][run_index]
            )
        else:
            client.log_metric(child_id, score_name, cv_results[score_name][run_index])
        client.log_metric(
            child_id,
            score_name.replace("mean", "std"),
            cv_results[score_name.replace("mean", "std", 1)][run_index],
        )

    print("Logging extra data related to the experiment")
    # client.set_tag(child_id, tags)

    run_id = child_run.info.run_uuid
    client.set_terminated(run_id)

    print("runID: %s" % run_id)


if __name__ == "__main__":
    np.random.seed(40)
    print("loading config")
    config = load_config("config.yaml")
    # Read the images's hand-crafted features from csv
    print("loading data")
    df = pd.read_csv(os.path.abspath("extracao/data.csv"))

    # Split the data into training and test sets. (0.75, 0.25) split.
    print("splitting data")
    features = config["Features"]
    X = shuffle(selectFeatures(features, df), random_state=10)
    target = config["Target"]
    y = shuffle(selectTarget(target, df), random_state=10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=50
    )

    y_train = np.array(y_train).flatten().astype(int)
    y_test = np.array(y_test).flatten().astype(int)

    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 4,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }

    models = config["Models"]
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    experiment_name = "GridSearchCV"

    client = MlflowClient(tracking_uri="http://localhost:5000")
    try:
        experiment_id = client.create_experiment(experiment_name)
    except:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    experiment = mlflow.set_experiment(experiment_name)
    print("starting training")
    for model in models:
        param_grid = load_grid_params(config, model)
        skmodel = sklearn_models[model]()
        for feature_set in powerset(features, 6, 6):
            feature_set = list(feature_set)
            X_train_fs = X_train[feature_set]
            X_test_fs = X_test[feature_set]

            tags = {"features": feature_set, "target": target}
            with mlflow.start_run(
                run_name=f"{model}_{len(feature_set)}_feats_CV"
            ) as run:
                print("Model:", model)
                _scoring = {
                    "r2",
                    "neg_mean_absolute_error",
                    "neg_root_mean_squared_error",
                }
                gsCV = GridSearchCV(
                    skmodel, param_grid, scoring=_scoring, refit="r2", verbose=2
                )
                training(gsCV, X_train_fs, y_train)
                # rmse, mae, r2 = eval_metrics_cv(gsCV.cv_results_)
                run_id = run.info.run_id

                for i in range(len(gsCV.cv_results_["params"])):
                    log_run(gsCV, model, i, tags, experiment_id, run_id)
                print("SCORE:")
                print(gsCV.score(X_test_fs, y_test))
                predicted_qualities = gsCV.predict(X_test_fs)
                (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

                # skmodel.fit(X_train_fs, y_train)
                # print("Training Complete. Predicting")
                # predicted_qualities = skmodel.predict(X_test_fs)
                # (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

                # results = cross_validation(
                #     skmodel, X_train_fs, np.array(y).flatten().astype(int), 8
                # )
                # rmse, mae, r2 = eval_metrics_cv(results)

                print("  RMSE: %s" % rmse)
                print("  MAE: %s" % mae)
                print("  R2: %s" % r2)
                params = {"model": model, "n_features": len(feature_set)}

                metrics = {
                    "mean_test_r2": r2,
                    "mean_test_neg_root_mean_squared_error": rmse,
                    "mean_test_neg_mean_absolute_error": mae,
                }

                mlflow_log(params=params, tags=tags, metrics=metrics, model=None)
                for k, v in tags.items():
                    mlflow.set_tag(k, v)

                print("Logging CV results matrix")
                tempdir = tempfile.TemporaryDirectory().name
                os.mkdir(tempdir)
                timestamp = (
                    datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
                )
                cv_results = gsCV.cv_results_

                filename = "%s-%s-cv_results.csv" % (model, timestamp)
                csv = os.path.join(tempdir, filename)
                pd.DataFrame(cv_results).to_csv(csv, index=False)

                mlflow.log_artifact(csv, "cv_results")

                # mlflow.log_param("model", model)
                # mlflow.log_param("n_features", len(feature_set))
                # mlflow.set_tag("features", feature_set)
                # mlflow.set_tag("target", target)
                # # mlflow.log_param("max_iter", max_iter)
                # mlflow.log_metric("rmse", rmse)
                # mlflow.log_metric("r2", r2)
                # mlflow.log_metric("mae", mae)

                # mlflow.sklearn.log_model(skmodel, f"{model}_{len(features)}_{runN}")

            mlflow.end_run()
