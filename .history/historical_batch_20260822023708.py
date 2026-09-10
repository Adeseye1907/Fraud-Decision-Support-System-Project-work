# ============================================================
# HISTORICAL BATCH FRAUD DETECTION
# + SHAP EXPLAINABILITY
# Runs once over the historical test dataset (known labels)
# and saves results for the Dashboard's Historical view.
# ============================================================

import json
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# 1. LOAD MODEL, SHAP EXPLAINER, HISTORICAL DATA
# ============================================================

rf_model = joblib.load("rf_fraud_model.pkl")
print("Fraud detection model loaded successfully.")

explainer = shap.TreeExplainer(rf_model)
print("SHAP explainer loaded successfully.")

historical_df = pd.read_csv("historical_test_data.csv")
print(f"Historical dataset loaded: {len(historical_df)} transactions.")

feature_columns = [c for c in historical_df.columns if c != "Class"]
X_hist = historical_df[feature_columns]
y_hist_true = historical_df["Class"]


# ============================================================
# 2. RUN PREDICTIONS ON THE FULL HISTORICAL BATCH
# ============================================================

y_hist_pred = rf_model.predict(X_hist)
y_hist_proba = rf_model.predict_proba(X_hist)[:, 1]

print("Predictions complete.")


# ============================================================
# 3. GLOBAL SHAP VALUES (sample for speed if dataset is large)
# ============================================================

SAMPLE_SIZE = min(1000, len(X_hist))
X_sample = X_hist.sample(SAMPLE_SIZE, random_state=42)

shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    fraud_shap_values = shap_values[1]
else:
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        fraud_shap_values = shap_values[:, :, 1]
    else:
        fraud_shap_values = shap_values

global_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Mean_Abs_SHAP": np.abs(fraud_shap_values).mean(axis=0)
}).sort_values("Mean_Abs_SHAP", ascending=False)

print("\n--- TOP 10 GLOBAL SHAP FEATURES ---")
print(global_importance.head(10).to_string(index=False))


# ============================================================
# 4. BUILD RESULTS TABLE (per-transaction, for the dashboard)
# ============================================================

results_df = historical_df.copy()
results_df["prediction"] = np.where(y_hist_pred == 1, "FRAUD", "LEGITIMATE")
results_df["fraud_probability"] = y_hist_proba
results_df["actual_label"] = np.where(y_hist_true == 1, "FRAUD", "LEGITIMATE")

results_df.to_csv("historical_predictions.csv", index=False)
print("\nSaved: historical_predictions.csv")


# ============================================================
# 5. SAVE GLOBAL SHAP IMPORTANCE (for dashboard charting)
# ============================================================

global_importance.to_csv("historical_global_shap.csv", index=False)
print("Saved: historical_global_shap.csv")


# ============================================================
# 6. SUMMARY METRICS (printed + saved as JSON)
# ============================================================

total = len(historical_df)
fraud_caught = int(((y_hist_true == 1) & (y_hist_pred == 1)).sum())
fraud_missed = int(((y_hist_true == 1) & (y_hist_pred == 0)).sum())
false_alarms = int(((y_hist_true == 0) & (y_hist_pred == 1)).sum())
actual_fraud = int((y_hist_true == 1).sum())

summary = {
    "total_transactions": total,
    "actual_fraud_count": actual_fraud,
    "fraud_caught": fraud_caught,
    "fraud_missed": fraud_missed,
    "false_alarms": false_alarms,
    "detection_rate": round(fraud_caught / actual_fraud, 4) if actual_fraud else 0,
}

print("\n--- HISTORICAL BATCH SUMMARY ---")
for k, v in summary.items():
    print(f"{k}: {v}")

with open("historical_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved: historical_summary.json")
print("\nHistorical batch analysis complete.")