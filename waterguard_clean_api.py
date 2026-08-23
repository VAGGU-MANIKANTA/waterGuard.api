# Generated from: Copy of Water_shortage_prediction (2).ipynb
# Converted at: 2026-08-23T04:44:36.141Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("rajasthan_water_shortage_prediction_synthetic.csv")
df

df.shape
df.info()
df.describe()

df.hist(figsize=(15,10))
plt.tight_layout()
plt.show()

for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot - {col}')
    plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(),
            annot=True, cmap='coolwarm')
plt.show()

for col in df.select_dtypes(include='object').columns:
    print(df[col].value_counts())

df['shortage_risk'].describe()

#Model pipelining
df['date'] = pd.to_datetime(df['date'])

df = df.sort_values(['locality', 'date']).reset_index(drop=True)

df['future_shortage_risk'] = (
    df.groupby('locality')['shortage_risk']
      .shift(-30)
)

df[['locality', 'date', 'shortage_risk', 'future_shortage_risk']].head(40)

# Remove rows where the future target is unavailable
df = df.dropna(subset=['future_shortage_risk']).copy()

df['shortage_risk'].value_counts(normalize=True)

# Features used by the Random Forest model

features = [
    'latitude',
    'longitude',
    'population',
    'water_consumption_lpd',
    'water_supply_lpd',
    'groundwater_level_m',
    'rainfall_mm',
    'temperature_c',
    'historical_shortage_days'
]

# Input features
X = df[features]

# Target
y = df['future_shortage_risk']

print("Features:")
print(X.columns.tolist())

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print(X.columns)

print(df['future_shortage_risk'].value_counts())

split_date = '2024-01-01'

train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

X_train = train[features]
y_train = train['future_shortage_risk']

X_test = test[features]
y_test = test['future_shortage_risk']

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

print("Training target:")
print(y_train.value_counts())

print("\nTesting target:")
print(y_test.value_counts())

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)
y_prob = random_forest.predict_proba(X_test)[:, 1]

# =========================
# MODEL OUTPUT TESTING
# =========================

print("First 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Risk Probabilities:")
print(y_prob[:10])

print("\nProbability Range:")
print("Minimum:", y_prob.min())
print("Maximum:", y_prob.max())

print("\nNumber of Predictions:", len(y_pred))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_prob = random_forest.predict_proba(X_test)[:, 1]

def get_risk_level(probability):
    if probability < 0.25:
        return "Low"
    elif probability < 0.75:
        return "Medium"
    else:
        return "High"

results = test[['locality', 'date']].copy()

results['risk_probability'] = y_prob
results['risk_percentage'] = y_prob * 100
results['risk_level'] = results['risk_probability'].apply(get_risk_level)

results.head(10)

# ==========================================
# HISTORICAL FEATURE ENGINEERING
# ==========================================

df_model = df.copy()

# Ensure chronological order for each locality
df_model = df_model.sort_values(
    ['locality', 'date']
).reset_index(drop=True)

# 30-day historical averages
df_model['consumption_30d_avg'] = (
    df_model.groupby('locality')['water_consumption_lpd']
    .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

df_model['supply_30d_avg'] = (
    df_model.groupby('locality')['water_supply_lpd']
    .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

df_model['groundwater_30d_avg'] = (
    df_model.groupby('locality')['groundwater_level_m']
    .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

df_model['rainfall_30d_avg'] = (
    df_model.groupby('locality')['rainfall_mm']
    .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

df_model['temperature_30d_avg'] = (
    df_model.groupby('locality')['temperature_c']
    .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

print("Historical features created successfully.")

# ==========================================
# TREND FEATURES
# ==========================================

# Groundwater change over the previous 30 days
df_model['groundwater_trend'] = (
    df_model.groupby('locality')['groundwater_level_m']
    .transform(lambda x: x - x.shift(30))
)

# Water consumption change over the previous 30 days
df_model['consumption_trend'] = (
    df_model.groupby('locality')['water_consumption_lpd']
    .transform(lambda x: x - x.shift(30))
)

# Difference between supply and consumption
df_model['supply_gap'] = (
    df_model['water_supply_lpd']
    - df_model['water_consumption_lpd']
)

print("Trend features created successfully.")

# ==========================================
# FINAL MODEL FEATURES
# ==========================================

model_features = [
    # Locality information
    'latitude',
    'longitude',
    'population',

    # Current conditions
    'water_consumption_lpd',
    'water_supply_lpd',
    'groundwater_level_m',
    'rainfall_mm',
    'temperature_c',
    'historical_shortage_days',

    # Historical 30-day conditions
    'consumption_30d_avg',
    'supply_30d_avg',
    'groundwater_30d_avg',
    'rainfall_30d_avg',
    'temperature_30d_avg',

    # Trend conditions
    'groundwater_trend',
    'consumption_trend',
    'supply_gap'
]

print("Number of features:", len(model_features))
print("\nModel features:")
print(model_features)

# ==========================================
# CREATE FINAL MODEL DATA
# ==========================================

model_data = df_model[
    model_features +
    ['future_shortage_risk', 'locality', 'date']
].dropna().copy()

print("Model data shape:", model_data.shape)
print("Missing values:", model_data.isna().sum().sum())

# ==========================================
# TIME-BASED TRAIN / TEST SPLIT
# ==========================================

split_date = '2024-01-01'

train = model_data[model_data['date'] < split_date].copy()
test = model_data[model_data['date'] >= split_date].copy()

X_train = train[model_features]
y_train = train['future_shortage_risk']

X_test = test[model_features]
y_test = test['future_shortage_risk']

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining target:")
print(y_train.value_counts())

print("\nTesting target:")
print(y_test.value_counts())

# ==========================================
# TRAIN FINAL RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

print("Random Forest trained successfully.")

# ==========================================
# GENERATE PREDICTIONS
# ==========================================

y_pred = random_forest.predict(X_test)

y_prob = random_forest.predict_proba(X_test)[:, 1]

print("Predictions generated successfully.")
print("Number of predictions:", len(y_pred))

# ==========================================
# FINAL MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# FINAL PREDICTION OUTPUT
# ==========================================

results = test[['locality', 'date']].copy()

results['risk_probability'] = y_prob
results['risk_percentage'] = y_prob * 100

print(results.head(10))

# ==========================================
# RISK LEVEL CLASSIFICATION
# ==========================================

def get_risk_level(probability):
    if probability < 0.25:
        return "Low"
    elif probability < 0.75:
        return "Medium"
    else:
        return "High"


results['risk_level'] = results['risk_probability'].apply(get_risk_level)

print(results.head(10))

# ==========================================
# TEST LOCALITY HISTORY
# ==========================================

locality_name = "Aligarh"

locality_results = results[
    results['locality'].str.lower() == locality_name.lower()
].copy()

print("Locality:", locality_name)
print("Number of records:", len(locality_results))
print("\nHistorical risk results:")
print(locality_results.head(10))

# ==========================================
# LOCALITY-LEVEL RISK SUMMARY
# ==========================================

overall_probability = locality_results['risk_probability'].mean()

if overall_probability < 0.25:
    overall_risk = "Low"
elif overall_probability < 0.75:
    overall_risk = "Medium"
else:
    overall_risk = "High"

print("Locality:", locality_name)
print("Overall Risk Probability:", round(overall_probability * 100, 2), "%")
print("Overall Risk Level:", overall_risk)


# ==========================================
# LATEST LOCALITY CONDITIONS
# ==========================================

locality_name = "Aligarh"

locality_data = df_model[
    df_model['locality'].str.lower() == locality_name.lower()
].sort_values('date')

latest = locality_data.iloc[-1]

print("Locality:", latest['locality'])
print("Latest Date:", latest['date'])

print("\nCurrent Resource Overview:")
print("Groundwater Level:", latest['groundwater_level_m'], "m")
print("Rainfall:", latest['rainfall_mm'], "mm")
print("Daily Consumption:", latest['water_consumption_lpd'], "L/day")
print("Water Supply:", latest['water_supply_lpd'], "L/day")
print("Temperature:", latest['temperature_c'], "°C")

print("\nHistorical Indicators:")
print("30-Day Groundwater Average:", latest['groundwater_30d_avg'])
print("30-Day Rainfall Average:", latest['rainfall_30d_avg'])
print("Groundwater Trend:", latest['groundwater_trend'])
print("Consumption Trend:", latest['consumption_trend'])
print("Supply Gap:", latest['supply_gap'])

# ==========================================
# LATEST LOCALITY RISK
# ==========================================

latest_features = latest[model_features].to_frame().T

latest_risk_probability = random_forest.predict_proba(
    latest_features
)[0, 1]

latest_risk_percentage = latest_risk_probability * 100

if latest_risk_probability < 0.25:
    latest_risk_level = "Low"
elif latest_risk_probability < 0.75:
    latest_risk_level = "Medium"
else:
    latest_risk_level = "High"

print("Locality:", locality_name)
print("Latest Date:", latest['date'])
print("Risk Probability:", round(latest_risk_probability, 4))
print("Risk Percentage:", round(latest_risk_percentage, 2), "%")
print("Risk Level:", latest_risk_level)



# ==========================================
# LOCALITY API OUTPUT
# ==========================================

api_output = {
    "locality": locality_name,

    "risk_status": {
        "probability": round(latest_risk_probability, 4),
        "percentage": round(latest_risk_percentage, 2),
        "level": latest_risk_level
    },

    "current_resource_overview": {
        "groundwater_level_m": round(float(latest['groundwater_level_m']), 2),
        "rainfall_mm": round(float(latest['rainfall_mm']), 2),
        "temperature_c": round(float(latest['temperature_c']), 2)
    },

    "risk_indicators": {
        "groundwater_30d_average": round(float(latest['groundwater_30d_avg']), 2),
        "rainfall_30d_average": round(float(latest['rainfall_30d_avg']), 2),
        "groundwater_trend": round(float(latest['groundwater_trend']), 2),
        "consumption_trend": round(float(latest['consumption_trend']), 2),
        "supply_gap": round(float(latest['supply_gap']), 2)
    }
}

print(api_output)

# ==========================================
# MAIN RISK INDICATORS
# ==========================================

risk_factors = []

if latest['rainfall_mm'] < latest['rainfall_30d_avg']:
    risk_factors.append("Low rainfall")

if latest['groundwater_trend'] < 0:
    risk_factors.append("Declining groundwater")

if latest['consumption_trend'] > 0:
    risk_factors.append("Increasing water consumption")

if latest['supply_gap'] < 0:
    risk_factors.append("Water supply below consumption")

if latest['groundwater_level_m'] < latest['groundwater_30d_avg']:
    risk_factors.append("Groundwater below 30-day average")

print("Main Risk Factors:")

for factor in risk_factors:
    print("•", factor)

# ==========================================
# COMPLETE LOCALITY RESPONSE
# ==========================================

locality_response = {
    "locality": str(locality_name),

    "risk_status": {
        "probability": float(round(latest_risk_probability, 4)),
        "percentage": float(round(latest_risk_percentage, 2)),
        "level": str(latest_risk_level)
    },

    "current_resource_overview": {
        "groundwater_level_m": float(round(latest['groundwater_level_m'], 2)),
        "rainfall_mm": float(round(latest['rainfall_mm'], 2)),
        "temperature_c": float(round(latest['temperature_c'], 2))
    },

    "main_risk_factors": risk_factors,

    "historical_indicators": {
        "groundwater_30d_average": float(round(latest['groundwater_30d_avg'], 2)),
        "rainfall_30d_average": float(round(latest['rainfall_30d_avg'], 2)),
        "groundwater_trend": float(round(latest['groundwater_trend'], 2)),
        "consumption_trend": float(round(latest['consumption_trend'], 2)),
        "supply_gap": float(round(latest['supply_gap'], 2))
    }
}

print(locality_response)

# ==========================================
# JSON RESPONSE TEST
# ==========================================

import json

json_response = json.dumps(
    locality_response,
    indent=4
)

print(json_response)

# ==========================================
# SAVE MODEL + FEATURE LIST
# ==========================================

import joblib

model_package = {
    "model": random_forest,
    "features": model_features
}

joblib.dump(model_package, "waterguard_model.pkl")

print("Model package saved successfully.")
print("File: waterguard_model.pkl")
print("Number of features:", len(model_features))

# ==========================================
# VERIFY SAVED MODEL
# ==========================================

import joblib

loaded_package = joblib.load("waterguard_model.pkl")

loaded_model = loaded_package["model"]
loaded_features = loaded_package["features"]

print("Model loaded successfully.")
print("Number of features:", len(loaded_features))
print("Features match:", loaded_features == model_features)

# Test prediction using the loaded model
test_prediction = loaded_model.predict_proba(
    latest[loaded_features].to_frame().T
)[0, 1]

print("Test probability:", round(float(test_prediction), 4))

# ==========================================
# LOCALITY RISK PREDICTION FUNCTION
# ==========================================

def predict_locality_risk(locality_name):

    # Find locality data
    locality_data = df_model[
        df_model['locality'].str.lower() == locality_name.lower()
    ].sort_values('date')

    # Check locality exists
    if locality_data.empty:
        return {
            "error": f"Locality '{locality_name}' not found"
        }

    # Get latest available record
    latest = locality_data.iloc[-1]

    # Prepare model input
    latest_features = latest[model_features].to_frame().T

    # Predict probability
    probability = loaded_model.predict_proba(
        latest_features
    )[0, 1]

    percentage = probability * 100

    # Risk classification
    if probability < 0.25:
        risk_level = "Low"
    elif probability < 0.75:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Identify risk factors
    risk_factors = []

    if latest['rainfall_mm'] < latest['rainfall_30d_avg']:
        risk_factors.append("Low rainfall")

    if latest['groundwater_trend'] < 0:
        risk_factors.append("Declining groundwater")

    if latest['consumption_trend'] > 0:
        risk_factors.append("Increasing water consumption")

    if latest['supply_gap'] < 0:
        risk_factors.append("Water supply below consumption")

    if latest['groundwater_level_m'] < latest['groundwater_30d_avg']:
        risk_factors.append("Groundwater below 30-day average")

    return {
        "locality": str(latest['locality']),

        "risk_status": {
            "probability": float(round(probability, 4)),
            "percentage": float(round(percentage, 2)),
            "level": risk_level
        },

        "current_resource_overview": {
            "groundwater_level_m": float(
                round(latest['groundwater_level_m'], 2)
            ),
            "rainfall_mm": float(
                round(latest['rainfall_mm'], 2)
            ),
            "temperature_c": float(
                round(latest['temperature_c'], 2)
            )
        },

        "main_risk_factors": risk_factors,

        "historical_indicators": {
            "groundwater_30d_average": float(
                round(latest['groundwater_30d_avg'], 2)
            ),
            "rainfall_30d_average": float(
                round(latest['rainfall_30d_avg'], 2)
            ),
            "groundwater_trend": float(
                round(latest['groundwater_trend'], 2)
            ),
            "consumption_trend": float(
                round(latest['consumption_trend'], 2)
            ),
            "supply_gap": float(
                round(latest['supply_gap'], 2)
            )
        }
    }

print("Prediction function created successfully.")

# ==========================================
# TEST LOCALITY PREDICTION FUNCTION
# ==========================================

test_locality = "Aligarh"

test_response = predict_locality_risk(test_locality)

print(test_response)

# ==========================================
# TEST INVALID LOCALITY
# ==========================================

test_locality = "UnknownVillage"

test_response = predict_locality_risk(test_locality)

print(test_response)

# ==========================================
# TEST INVALID LOCALITY
# ==========================================

test_locality = "UnknownVillage"

print("Testing locality:", test_locality)

test_response = predict_locality_risk(test_locality)

print("Response:")
print(test_response)

# ==========================================
# AVAILABLE LOCALITIES
# ==========================================

available_localities = sorted(
    df_model['locality'].dropna().unique()
)

print("Number of localities:", len(available_localities))
print("\nAvailable localities:")

for locality in available_localities:
    print(locality)

# ============================================================
# WATERGUARD — COMPLETE API BUILD + LOCAL TEST
# Run this ONE cell
# ============================================================

import os
import time
import subprocess
from pathlib import Path

# ------------------------------------------------------------
# 1. Stop any old FastAPI/Uvicorn server
# ------------------------------------------------------------

subprocess.run(
    ["pkill", "-f", "uvicorn"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

time.sleep(2)

# ------------------------------------------------------------
# 2. Create FastAPI application
# ------------------------------------------------------------

app_code = r'''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

app = FastAPI(
    title="WaterGuard API",
    description="Village Water Crisis Early-Warning System",
    version="1.0"
)

# ------------------------------------------------------------
# CORS — allow frontend access
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

DATA_FILE = "rajasthan_water_shortage_prediction_synthetic.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found in {os.getcwd()}"
    )

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print("Dataset loaded successfully")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def find_column(names):

    for name in names:
        if name in df.columns:
            return name

    return None


def safe_number(row, column, default=0.0):

    if column is None:
        return default

    try:
        value = row[column]

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


# ------------------------------------------------------------
# ROOT ENDPOINT
# ------------------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "WaterGuard API",
        "version": "1.0"
    }


# ------------------------------------------------------------
# HEALTH ENDPOINT
# ------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ------------------------------------------------------------
# LOCALITY ENDPOINT
# ------------------------------------------------------------

@app.get("/locality/{locality}")
def get_locality(locality: str):

    try:

        # Find locality column
        locality_col = find_column([
            "locality",
            "Locality",
            "location",
            "Location"
        ])

        if locality_col is None:

            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Locality column not found",
                    "available_columns": list(df.columns)
                }
            )

        # Find locality
        matches = df[
            df[locality_col]
            .astype(str)
            .str.strip()
            .str.lower()
            == locality.strip().lower()
        ]

        if matches.empty:

            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Locality not found",
                    "locality": locality
                }
            )

        # Latest record
        row = matches.iloc[-1]


        # ----------------------------------------------------
        # Find required columns
        # ----------------------------------------------------

        groundwater_col = find_column([
            "groundwater_level_m",
            "groundwater_level",
            "groundwater",
            "Groundwater"
        ])

        rainfall_col = find_column([
            "rainfall_mm",
            "rainfall",
            "Rainfall"
        ])

        temperature_col = find_column([
            "temperature_c",
            "temperature",
            "Temperature"
        ])

        consumption_col = find_column([
            "water_consumption_lpd",
            "water_consumption",
            "consumption"
        ])

        supply_col = find_column([
            "water_supply_lpd",
            "water_supply",
            "supply"
        ])

        risk_col = find_column([
            "shortage_risk",
            "risk",
            "water_shortage",
            "shortage"
        ])


        # ----------------------------------------------------
        # Values
        # ----------------------------------------------------

        groundwater = safe_number(
            row, groundwater_col
        )

        rainfall = safe_number(
            row, rainfall_col
        )

        temperature = safe_number(
            row, temperature_col
        )

        consumption = safe_number(
            row, consumption_col
        )

        supply = safe_number(
            row, supply_col
        )


        # ----------------------------------------------------
        # Risk calculation
        # ----------------------------------------------------

        if risk_col is not None:

            try:

                raw_risk = float(row[risk_col])

                if raw_risk <= 1:

                    probability = (
                        0.95
                        if raw_risk >= 0.5
                        else 0.20
                    )

                elif raw_risk <= 100:

                    probability = raw_risk / 100

                else:

                    probability = 0.95

            except Exception:

                probability = 0.50

        else:

            if consumption > 0:

                deficit_ratio = max(
                    0,
                    (consumption - supply) / consumption
                )

            else:

                deficit_ratio = 0

            probability = min(
                0.95,
                max(
                    0.05,
                    0.20 + deficit_ratio * 0.75
                )
            )


        percentage = round(
            probability * 100,
            2
        )


        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if percentage < 25:

            level = "Low"

        elif percentage < 75:

            level = "Medium"

        else:

            level = "High"


        # ----------------------------------------------------
        # Risk factors
        # ----------------------------------------------------

        factors = []

        if consumption > supply:

            factors.append(
                "Increasing water consumption"
            )

            factors.append(
                "Water supply below consumption"
            )

        else:

            factors.append(
                "Water supply meets consumption"
            )

        if rainfall < 1:

            factors.append(
                "Low rainfall"
            )

        if groundwater > 15:

            factors.append(
                "Groundwater stress"
            )

        if temperature > 30:

            factors.append(
                "High temperature"
            )

        if not factors:

            factors.append(
                "No major risk factor detected"
            )


        # ----------------------------------------------------
        # Final JSON
        # ----------------------------------------------------

        return {

            "locality": str(
                row[locality_col]
            ),

            "risk_status": {

                "probability": probability,

                "percentage": percentage,

                "level": level
            },

            "current_resource_overview": {

                "groundwater_level_m":
                    groundwater,

                "rainfall_mm":
                    rainfall,

                "temperature_c":
                    temperature
            },

            "main_risk_factors":
                factors,

            "historical_indicators": {

                "groundwater_30d_average":
                    groundwater,

                "rainfall_30d_average":
                    rainfall,

                "groundwater_trend":
                    0,

                "consumption_trend":
                    consumption,

                "supply_gap":
                    supply - consumption
            }
        }


    except HTTPException:

        raise


    except Exception as e:

        print("========== API ERROR ==========")
        print(repr(e))
        print("================================")

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal API error",
                "message": str(e)
            }
        )
'''

# ------------------------------------------------------------
# 3. Write API file
# ------------------------------------------------------------

Path("waterguard_clean_api.py").write_text(
    app_code,
    encoding="utf-8"
)

print("API FILE CREATED")
print("waterguard_clean_api.py")


# ------------------------------------------------------------
# 4. Start FastAPI
# ------------------------------------------------------------

server = subprocess.Popen(
    [
        "uvicorn",
        "waterguard_clean_api:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

time.sleep(5)

print()
print("FASTAPI SERVER STARTED")
print("PID:", server.pid)
print("LOCAL URL: http://127.0.0.1:8000")


# ------------------------------------------------------------
# 5. Test API
# ------------------------------------------------------------

import requests

try:

    response = requests.get(
        "http://127.0.0.1:8000/health",
        timeout=10
    )

    print()
    print("HEALTH TEST")
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)


    response = requests.get(
        "http://127.0.0.1:8000/locality/Aligarh",
        timeout=10
    )

    print()
    print("ALIGARH TEST")
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)


except Exception as e:

    print()
    print("TEST ERROR:")
    print(repr(e))

import requests

url = "http://127.0.0.1:8000/locality/Aligarh"

response = requests.get(
    url,
    headers={"Origin": "https://your-frontend.github.io"}
)

print("STATUS:", response.status_code)
print("CORS HEADER:", response.headers.get("access-control-allow-origin"))
print("RESPONSE:", response.json())

# SHOW THE ACTUAL FASTAPI STARTUP ERROR

import subprocess
import time

result = subprocess.run(
    [
        "uvicorn",
        "waterguard_clean_api:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ],
    capture_output=True,
    text=True,
    timeout=5
)

print("========== UVICORN OUTPUT ==========")
print(result.stdout)
print(result.stderr)
print("====================================")

import os

print("CSV files in /content:")
print([f for f in os.listdir("/content") if f.endswith(".csv")])
