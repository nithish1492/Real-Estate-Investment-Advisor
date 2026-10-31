
"""Leakage-aware training, calibration, independent test evaluation and MLflow tracking.

Run from project root:
    python -m src.train

Optional quick run:
    python -m src.train --sample 50000
"""

import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier, XGBRegressor

from . import config as C
from . import features as F


FEATURES = (
    C.NUMERIC_FEATURES
    + C.BINARY_FEATURES
    + C.CATEGORICAL_FEATURES
)


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                C.NUMERIC_FEATURES,
            ),
            (
                "bin",
                "passthrough",
                C.BINARY_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                ),
                C.CATEGORICAL_FEATURES,
            ),
        ]
    )


# ---------------------------------------------------------
# CLASSIFICATION MODELS
# ---------------------------------------------------------

def clf_models():

    rs = C.RANDOM_STATE

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=300,
            C=1.0,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=5,
            max_samples=0.5,
            n_jobs=-1,
            random_state=rs,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=rs,
        ),
    }


# ---------------------------------------------------------
# REGRESSION MODELS
# ---------------------------------------------------------

def reg_models():

    rs = C.RANDOM_STATE

    return {
        "Ridge": Ridge(
            alpha=1.0,
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=5,
            max_samples=0.5,
            n_jobs=-1,
            random_state=rs,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=rs,
        ),
    }


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

def feature_importance(pipe):

    model = pipe.named_steps["model"]

    names = pipe.named_steps["prep"].get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = np.abs(np.ravel(model.coef_))

    agg = {}

    for name, value in zip(names, imp):

        kind, rest = name.split("__", 1)

        original = rest

        if kind == "cat":

            for col in C.CATEGORICAL_FEATURES:

                if rest.startswith(col + "_"):
                    original = col
                    break

        agg[original] = agg.get(original, 0.0) + float(value)

    values = pd.Series(agg)

    total = values.sum()

    if total > 0:
        values = values / total

    return (
        values.sort_values(ascending=False)
        .rename("importance")
        .to_frame()
    )


# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------

def prepare_data(raw_part, reference=None):

    part = F.clean(raw_part)

    part = F.add_row_features(part)

    if reference is None:

        reference = F.build_reference_tables(part)

        reference["growth_median"] = float(
            F.growth_rate(part).median()
        )

    part = F.add_market_features(
        part,
        reference,
    )

    part = F.make_targets(
        part,
        reference["growth_median"],
        reference["infra_median"],
    )

    return part, reference


# ---------------------------------------------------------
# CLASSIFICATION METRICS
# ---------------------------------------------------------

def classification_metrics(y_true, predictions, probabilities):

    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7,
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
        ).tolist(),
    }


# ---------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------

def main(sample=None):

    started = time.time()

    C.MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading dataset...")

    raw = pd.read_csv(C.RAW_CSV)

    raw.columns = [
        col.strip()
        for col in raw.columns
    ]

    raw = raw.drop_duplicates(
        subset=[
            col
            for col in raw.columns
            if col != "ID"
        ]
    ).reset_index(drop=True)

    if sample and sample < len(raw):

        raw = raw.sample(
            n=sample,
            random_state=C.RANDOM_STATE,
        ).reset_index(drop=True)

    print("Total dataset rows:", len(raw))

    # -----------------------------------------------------
    # FOUR INDEPENDENT DATA PARTITIONS
    # -----------------------------------------------------

    print("\nSplitting dataset...")

    # 60% training, 40% temporary holdout
    fit_raw, hold_raw = train_test_split(
        raw,
        test_size=0.40,
        random_state=C.RANDOM_STATE,
    )

    # 15% validation, 25% remaining
    val_raw, hold_raw = train_test_split(
        hold_raw,
        test_size=0.625,
        random_state=C.RANDOM_STATE,
    )

    # 10% calibration, 15% final test
    cal_raw, test_raw = train_test_split(
        hold_raw,
        test_size=0.60,
        random_state=C.RANDOM_STATE,
    )

    print("Preparing training data...")

    # Build reference statistics using training data only.
    fit, reference = prepare_data(fit_raw)

    # Use the training reference statistics for all other partitions.
    val, _ = prepare_data(val_raw, reference)
    cal, _ = prepare_data(cal_raw, reference)
    test, _ = prepare_data(test_raw, reference)

    # Save training-only reference statistics.
    with open(
        C.MODEL_DIR / "reference_tables.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            reference,
            file,
            indent=2,
        )

    # Save combined processed data for the existing dashboard.
    combined = pd.concat(
        [fit, val, cal, test],
        ignore_index=True,
    )

    combined.to_csv(
        C.CLEAN_CSV,
        index=False,
    )

    # -----------------------------------------------------
    # FEATURES AND TARGETS
    # -----------------------------------------------------

    X_fit = fit[FEATURES]
    X_val = val[FEATURES]
    X_cal = cal[FEATURES]
    X_test = test[FEATURES]

    yc_fit = fit["Good_Investment"]
    yc_val = val["Good_Investment"]
    yc_cal = cal["Good_Investment"]
    yc_test = test["Good_Investment"]

    yr_fit = fit["Future_Price_5Y_Lakhs"]
    yr_val = val["Future_Price_5Y_Lakhs"]
    yr_test = test["Future_Price_5Y_Lakhs"]

    print("\nDataset partition sizes:")

    print("Training:", len(fit))
    print("Validation:", len(val))
    print("Calibration:", len(cal))
    print("Final test:", len(test))

    # -----------------------------------------------------
    # MLFLOW INITIALIZATION
    # -----------------------------------------------------

    mlflow.set_tracking_uri(C.MLFLOW_URI)

    mlflow.set_experiment(
        C.MLFLOW_EXPERIMENT
    )

    results = {
        "classification": {},
        "regression": {},
    }

    # -----------------------------------------------------
    # CLASSIFICATION TRAINING
    # -----------------------------------------------------

    print("\n========== CLASSIFICATION ==========")

    classifier_candidates = {}

    best_classifier_name = None
    best_classifier_score = -1

    for name, estimator in clf_models().items():

        print("\nTraining classifier:", name)

        pipeline = Pipeline(
            steps=[
                ("prep", preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(
            X_fit,
            yc_fit,
        )

        validation_probability = (
            pipeline.predict_proba(X_val)[:, 1]
        )

        validation_prediction = (
            validation_probability >= 0.5
        ).astype(int)

        metrics = classification_metrics(
            yc_val,
            validation_prediction,
            validation_probability,
        )

        classifier_candidates[name] = pipeline

        results["classification"][name] = {
            "validation_" + key: value
            for key, value in metrics.items()
            if key != "confusion_matrix"
        }

        print(
            f"{name} validation F1: "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"{name} validation ROC-AUC: "
            f"{metrics['roc_auc']:.4f}"
        )

        if metrics["f1"] > best_classifier_score:

            best_classifier_score = metrics["f1"]

            best_classifier_name = name

    # -----------------------------------------------------
    # REGRESSION TRAINING
    # -----------------------------------------------------

    print("\n========== REGRESSION ==========")

    regression_candidates = {}

    best_regressor_name = None
    best_regressor_score = float("inf")

    for name, estimator in reg_models().items():

        print("\nTraining regressor:", name)

        pipeline = Pipeline(
            steps=[
                ("prep", preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(
            X_fit,
            yr_fit,
        )

        validation_prediction = pipeline.predict(
            X_val
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    yr_val,
                    validation_prediction,
                )
            )
        )

        mae = float(
            mean_absolute_error(
                yr_val,
                validation_prediction,
            )
        )

        r2 = float(
            r2_score(
                yr_val,
                validation_prediction,
            )
        )

        regression_candidates[name] = pipeline

        results["regression"][name] = {
            "validation_rmse": rmse,
            "validation_mae": mae,
            "validation_r2": r2,
        }

        print(
            f"{name} validation RMSE: {rmse:.4f}"
        )

        if rmse < best_regressor_score:

            best_regressor_score = rmse

            best_regressor_name = name

    # -----------------------------------------------------
    # CLASSIFIER CALIBRATION
    # -----------------------------------------------------

    print("\n========== CALIBRATION ==========")

    base_classifier = classifier_candidates[
        best_classifier_name
    ]

    calibrated_classifier = CalibratedClassifierCV(
        FrozenEstimator(base_classifier),
        method="sigmoid",
    )

    calibrated_classifier.fit(
        X_cal,
        yc_cal,
    )

    print(
        "Selected classifier:",
        best_classifier_name,
    )

    print("Calibration method: sigmoid")

    # -----------------------------------------------------
    # FINAL CLASSIFICATION TEST
    # -----------------------------------------------------

    print("\n========== FINAL CLASSIFICATION TEST ==========")

    test_probability = (
        calibrated_classifier.predict_proba(X_test)[:, 1]
    )

    test_prediction = (
        test_probability >= 0.5
    ).astype(int)

    final_classifier_metrics = classification_metrics(
        yc_test,
        test_prediction,
        test_probability,
    )

    results["classification"][best_classifier_name].update(
        final_classifier_metrics
    )

    results["classification"]["_champion"] = (
        best_classifier_name
    )

    print("\nFinal classification metrics:")

    for key, value in final_classifier_metrics.items():

        if key != "confusion_matrix":
            print(f"{key}: {value:.4f}")

    # -----------------------------------------------------
    # FINAL REGRESSION TEST
    # -----------------------------------------------------

    print("\n========== FINAL REGRESSION TEST ==========")

    best_regressor = regression_candidates[
        best_regressor_name
    ]

    regression_prediction = best_regressor.predict(
        X_test
    )

    baseline_prediction = (
        X_test["Price_in_Lakhs"]
        * (1 + C.BASE_GROWTH_RATE)
        ** C.HORIZON_YEARS
    )

    baseline_rmse = float(
        np.sqrt(
            mean_squared_error(
                yr_test,
                baseline_prediction,
            )
        )
    )

    baseline_mae = float(
        mean_absolute_error(
            yr_test,
            baseline_prediction,
        )
    )

    baseline_r2 = float(
        r2_score(
            yr_test,
            baseline_prediction,
        )
    )

    final_rmse = float(
        np.sqrt(
            mean_squared_error(
                yr_test,
                regression_prediction,
            )
        )
    )

    final_mae = float(
        mean_absolute_error(
            yr_test,
            regression_prediction,
        )
    )

    final_r2 = float(
        r2_score(
            yr_test,
            regression_prediction,
        )
    )

    results["regression"]["Baseline_Fixed_8pct"] = {
        "rmse": baseline_rmse,
        "mae": baseline_mae,
        "r2": baseline_r2,
    }

    results["regression"][best_regressor_name].update({
        "rmse": final_rmse,
        "mae": final_mae,
        "r2": final_r2,
    })

    results["regression"]["_champion"] = (
        best_regressor_name
    )

    print("\nRegression results:")

    print("Selected model:", best_regressor_name)
    print("Model RMSE:", round(final_rmse, 4))
    print("Model MAE:", round(final_mae, 4))
    print("Model R2:", round(final_r2, 4))

    print("\nFixed 8% baseline:")

    print("Baseline RMSE:", round(baseline_rmse, 4))
    print("Baseline MAE:", round(baseline_mae, 4))
    print("Baseline R2:", round(baseline_r2, 4))

    # -----------------------------------------------------
    # SAVE MODEL ARTIFACTS
    # -----------------------------------------------------

    print("\n========== SAVING MODELS ==========")

    joblib.dump(
        calibrated_classifier,
        C.MODEL_DIR / "classification_model.joblib",
        compress=3,
    )

    joblib.dump(
        best_regressor,
        C.MODEL_DIR / "regression_model.joblib",
        compress=3,
    )

    joblib.dump(
        calibrated_classifier,
        C.MODEL_DIR / "classification_calibrated_model.joblib",
        compress=3,
    )

    feature_importance(
        base_classifier
    ).to_csv(
        C.MODEL_DIR / "classification_feature_importance.csv"
    )

    feature_importance(
        best_regressor
    ).to_csv(
        C.MODEL_DIR / "regression_feature_importance.csv"
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = final_classifier_metrics["confusion_matrix"]

    fig, ax = plt.subplots(
        figsize=(4.5, 4)
    )

    ConfusionMatrixDisplay(
        np.array(cm),
        display_labels=[
            "Not good",
            "Good",
        ],
    ).plot(
        ax=ax,
        colorbar=False,
    )

    ax.set_title(
        f"{best_classifier_name} calibrated final-test confusion matrix"
    )

    fig.tight_layout()

    fig.savefig(
        C.MODEL_DIR / "final_confusion_matrix.png",
        dpi=140,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # MLFLOW CLASSIFICATION LOGGING
    # -----------------------------------------------------

    print("\nLogging classification to MLflow...")

    with mlflow.start_run(
        run_name=f"final_calibrated_{best_classifier_name}"
    ):

        mlflow.set_tags({
            "task": "classification",
            "stage": "final_test",
            "calibrated": "sigmoid",
        })

        mlflow.log_params({
            "model": best_classifier_name,
            "fit_rows": len(fit),
            "validation_rows": len(val),
            "calibration_rows": len(cal),
            "final_test_rows": len(test),
        })

        mlflow.log_metrics({
            key: value
            for key, value in final_classifier_metrics.items()
            if isinstance(value, (int, float))
        })

        mlflow.sklearn.log_model(
            calibrated_classifier,
            name="model",
            registered_model_name="RealEstate_GoodInvestment_Classifier",
            serialization_format="cloudpickle",
            input_example=X_test.head(3),
        )

    # -----------------------------------------------------
    # MLFLOW REGRESSION LOGGING
    # -----------------------------------------------------

    print("\nLogging regression to MLflow...")

    with mlflow.start_run(
        run_name=f"final_{best_regressor_name}"
    ):

        mlflow.set_tags({
            "task": "regression",
            "stage": "final_test",
        })

        mlflow.log_params({
            "model": best_regressor_name,
            "fit_rows": len(fit),
            "validation_rows": len(val),
            "final_test_rows": len(test),
        })

        mlflow.log_metrics({
            key: value
            for key, value in results["regression"][
                best_regressor_name
            ].items()
            if isinstance(value, (int, float))
        })

        mlflow.sklearn.log_model(
            best_regressor,
            name="model",
            registered_model_name="RealEstate_FuturePrice_Regressor",
            serialization_format="cloudpickle",
            input_example=X_test.head(3),
        )

    # -----------------------------------------------------
    # SAVE METRICS JSON
    # -----------------------------------------------------

    results["meta"] = {
        "fit_rows": len(fit),
        "validation_rows": len(val),
        "calibration_rows": len(cal),
        "test_rows": len(test),

        "positive_rate": float(
            yc_fit.mean()
        ),

        "split_strategy": (
            "60% fit / 15% validation / "
            "10% calibration / 15% independent final test"
        ),

        "calibration_method": "sigmoid (Platt scaling)",

        "seconds": round(
            time.time() - started,
            1,
        ),

        "target_warning": (
            "Targets are rule-generated from this dataset. "
            "Scores measure recovery of those rules, not "
            "real-world investment or resale-price accuracy."
        ),
    }

    with open(
        C.MODEL_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # -----------------------------------------------------
    # COMPLETION
    # -----------------------------------------------------

    print("\n========================================")
    print("       TRAINING COMPLETED")
    print("========================================")

    print(
        "Best classifier:",
        best_classifier_name,
    )

    print(
        "Best regressor:",
        best_regressor_name,
    )

    print(
        "Total time:",
        round(time.time() - started, 2),
        "seconds",
    )

    print("\nSaved files:")

    print("models/classification_model.joblib")
    print("models/regression_model.joblib")
    print("models/reference_tables.json")
    print("models/metrics.json")

    print("\nYour models are ready for testing.")


# ---------------------------------------------------------
# PROGRAM ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional smaller dataset sample for a quick test.",
    )

    args = parser.parse_args()

    main(args.sample)
