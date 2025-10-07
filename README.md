# Custom Dataset Predictor

A machine learning web app that lets you upload **any CSV file**, automatically trains a model on it, and lets you make predictions instantly.

Built as **Project 4** (the final project) of a 4-project Machine Learning series using the [Zero to Mastery ML course](https://github.com/mrdbourke/zero-to-mastery-ml) as a learning foundation.

---

## What It Does

1. You upload **any CSV file**
2. You pick the column you want to **predict** (target)
3. The app automatically:
   - Detects if it's a **classification** or **regression** problem
   - Fills in any missing values
   - Converts text columns to numbers
   - Trains a **Random Forest** model
   - Shows you the accuracy or R² score
4. You fill in values and get a **prediction instantly**

---

## How It Works (Auto-Detection)

| Target Column Type | Problem Type | Model Used |
|---|---|---|
| Text (e.g. "Yes"/"No") | Classification | RandomForestClassifier |
| Few unique numbers (≤ 10) | Classification | RandomForestClassifier |
| Many unique numbers | Regression | RandomForestRegressor |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Programming language |
| Scikit-Learn | ML model, Pipeline, Imputer, Encoder |
| Pandas & NumPy | Data handling |
| Flask | Web server |
| HTML & CSS | Multi-step frontend UI |
| Joblib | Saving/loading the model |

---

## How to Run

### 1. Install dependencies
```bash
pip install flask scikit-learn pandas numpy joblib
```

### 2. Start the web app
```bash
python3 app.py
```

### 3. Open in browser
```
http://127.0.0.1:5000
```

### 4. Upload a CSV, pick your target, and predict!

---

## Project Files

```
custom-dataset-predictor/
├── app.py                  # Flask app — all routes and ML logic
├── templates/
│   ├── upload.html         # Step 1: Upload CSV
│   ├── configure.html      # Step 2: Pick target column
│   └── result.html         # Step 3+4: Model metrics + prediction form
└── static/
    └── style.css           # Purple-themed styling
```

---

## Example Datasets to Try

| Dataset | Target Column | Type |
|---|---|---|
| [Heart Disease](https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv) | target | Classification |
| [Car Sales](https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales-extended.csv) | Price | Regression |
| Titanic | Survived | Classification |
| Iris | species | Classification |

---

## Key ML Concepts Used

- **Auto problem detection** — classifies the task based on the target column
- **Pipeline** — chains preprocessing and model into one object
- **SimpleImputer** — fills missing values automatically
- **OneHotEncoder** — converts text columns to numbers
- **ColumnTransformer** — applies different transformations to different columns
- **RandomForestClassifier / Regressor** — works for both problem types

---

## Part of ML Projects Series

| Project | Description | Repo |
|---|---|---|
| Project 1 | Heart Disease Predictor | [Link](https://github.com/manny2341/heart-disease-predictor) |
| Project 2 | Car Price Predictor | [Link](https://github.com/manny2341/car-price-predictor) |
| Project 3 | House Price Predictor | [Link](https://github.com/manny2341/house-price-predictor) |
| Project 4 | Custom Dataset Predictor | This repo |
