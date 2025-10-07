"""
Project 4: Custom Dataset Predictor - Flask Web App
Upload any CSV, pick a target column, train a model, make predictions.
Run with: python3 app.py
Then open: http://127.0.0.1:5000
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

UPLOAD_PATH  = os.path.join("uploads", "data.csv")
MODEL_PATH   = "model.joblib"
INFO_PATH    = "model_info.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_problem_type(series):
    """Classify or Regress based on target column."""
    unique = series.nunique()
    if series.dtype == object or unique <= 10:
        return "classification"
    return "regression"


def build_pipeline(X, problem_type):
    """Build a preprocessing + model pipeline based on the data."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="mean"), num_cols))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    estimator = (
        RandomForestClassifier(n_estimators=100, random_state=42)
        if problem_type == "classification"
        else RandomForestRegressor(n_estimators=100, random_state=42)
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    """Step 1 — Upload CSV."""
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Step 2 — Receive CSV, show column selector."""
    file = request.files.get("csvfile")
    if not file or not file.filename.endswith(".csv"):
        return render_template("upload.html", error="Please upload a valid .csv file.")

    file.save(UPLOAD_PATH)

    try:
        df = pd.read_csv(UPLOAD_PATH)
    except Exception as e:
        return render_template("upload.html", error=f"Could not read CSV: {e}")

    if df.shape[1] < 2:
        return render_template("upload.html", error="CSV must have at least 2 columns.")

    return render_template(
        "configure.html",
        columns=df.columns.tolist(),
        shape=df.shape,
        preview=df.head(5).to_html(classes="preview-table", index=False),
    )


@app.route("/train", methods=["POST"])
def train():
    """Step 3 — Train the model, save it, show metrics."""
    target_col = request.form.get("target")

    try:
        df = pd.read_csv(UPLOAD_PATH)
    except Exception as e:
        return redirect(url_for("home"))

    if target_col not in df.columns:
        return redirect(url_for("home"))

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Auto-detect problem type
    problem_type = detect_problem_type(y)

    # Encode y for classification if it's text
    y_labels = None
    if problem_type == "classification" and y.dtype == object:
        y_labels = y.unique().tolist()
        y = y.astype("category").cat.codes

    # Train / test split
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build and train pipeline
    pipeline = build_pipeline(X, problem_type)
    pipeline.fit(X_train, y_train)
    y_preds = pipeline.predict(X_test)

    # Metrics
    if problem_type == "classification":
        metric_name  = "Accuracy"
        metric_value = f"{accuracy_score(y_test, y_preds) * 100:.2f}%"
    else:
        metric_name  = "R² Score"
        metric_value = f"{r2_score(y_test, y_preds):.4f}"
        mae = mean_absolute_error(y_test, y_preds)

    # Save model + info
    dump(pipeline, MODEL_PATH)

    feature_cols = X.columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Collect unique values for categorical dropdowns
    cat_options = {col: sorted(df[col].dropna().unique().tolist()) for col in cat_cols}

    # Build quick-fill sample rows from the dataset
    sample_rows = {}
    if problem_type == "classification":
        for label_val in df[target_col].unique()[:4]:  # up to 4 classes
            row = df[df[target_col] == label_val][feature_cols].iloc[0]
            sample_rows[f"{target_col} = {label_val}"] = {
                k: (round(float(v), 4) if isinstance(v, float) else v)
                for k, v in row.items()
            }
    else:
        # For regression: show lowest, median, highest priced samples
        for label, row in [
            ("Low value",    df.nsmallest(1, target_col)[feature_cols].iloc[0]),
            ("Mid value",    df.iloc[len(df)//2][feature_cols]),
            ("High value",   df.nlargest(1, target_col)[feature_cols].iloc[0]),
        ]:
            sample_rows[label] = {
                k: (round(float(v), 4) if isinstance(v, float) else v)
                for k, v in row.items()
            }

    info = {
        "target": target_col,
        "problem_type": problem_type,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_options": cat_options,
        "y_labels": y_labels,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "mae": f"{mae:.4f}" if problem_type == "regression" else None,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sample_rows": sample_rows,
    }

    with open(INFO_PATH, "w") as f:
        json.dump(info, f)

    return render_template("result.html", info=info)


@app.route("/predict", methods=["POST"])
def predict():
    """Step 4 — Make a prediction from user inputs."""
    try:
        pipeline = load(MODEL_PATH)
        with open(INFO_PATH) as f:
            info = json.load(f)
    except Exception:
        return redirect(url_for("home"))

    # Build input row from form
    row = {}
    for col in info["feature_cols"]:
        val = request.form.get(col, "")
        if col in info["num_cols"]:
            try:
                row[col] = float(val)
            except ValueError:
                row[col] = np.nan
        else:
            row[col] = val if val else np.nan

    input_df = pd.DataFrame([row])

    prediction = pipeline.predict(input_df)[0]

    # Convert numeric code back to label for classification
    if info["problem_type"] == "classification" and info["y_labels"]:
        try:
            prediction = info["y_labels"][int(prediction)]
        except (IndexError, ValueError):
            prediction = str(prediction)

    # Probability for classification
    confidence = None
    if info["problem_type"] == "classification":
        try:
            proba = pipeline.predict_proba(input_df)[0]
            confidence = f"{max(proba) * 100:.1f}%"
        except Exception:
            pass

    return render_template(
        "result.html",
        info=info,
        prediction=str(prediction),
        confidence=confidence,
        user_input=row,
    )


if __name__ == "__main__":
    app.run(debug=True)
