# ============================================================
# ADAPTIVE FRAUD DETECTION MONITORING
# ============================================================
#
# Development of Adaptive and Interpretable Machine Learning
# Framework for Real-Time Fraud Detection and Operational
# Decision Support in Digital Banking
#
# Purpose:
#   1. Establish a reference distribution from historical data
#   2. Compare current transaction data against the reference
#   3. Calculate Population Stability Index (PSI)
#   4. Detect potential data / behaviour drift
#   5. Produce an adaptive monitoring report
#
# Dataset:
#   creditcard.csv
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import json
import numpy as np
import pandas as pd


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_FILE = "creditcard dataset.csv"

DRIFT_REPORT_FILE = "adaptive_drift_report.csv"

DRIFT_SUMMARY_FILE = "adaptive_drift_summary.json"

RANDOM_STATE = 42


# ============================================================
# 3. MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 60)
print("ADAPTIVE FRAUD DETECTION MONITORING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded successfully.")
print(f"Total observations: {len(df):,}")


# ============================================================
# 5. CHECK REQUIRED FEATURES
# ============================================================

missing_columns = [
    col
    for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"The following required features are missing: "
        f"{missing_columns}"
    )


# ============================================================
# 6. REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=FEATURE_COLUMNS
).copy()

print(
    f"Observations after removing missing values: "
    f"{len(df):,}"
)


# ============================================================
# 7. CREATE REFERENCE AND CURRENT DATA
# ============================================================
#
# Reference data represents the historical environment
# in which the model was developed.
#
# Current data represents a later/current transaction
# environment that we want to monitor for change.
#
# IMPORTANT:
# We are not retraining the model here.
# We are monitoring whether the environment has changed.
# ============================================================

REFERENCE_SIZE = int(len(df) * 0.70)

reference_df = df.iloc[
    :REFERENCE_SIZE
].copy()

current_df = df.iloc[
    REFERENCE_SIZE:
].copy()

print(
    f"\nReference observations: {len(reference_df):,}"
)

print(
    f"Current observations: {len(current_df):,}"
)


# ============================================================
# 8. PSI FUNCTION
# ============================================================

def calculate_psi(
    reference,
    current,
    bins=10
):

    reference = pd.Series(
        reference
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    current = pd.Series(
        current
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if reference.empty or current.empty:

        return np.nan

    # --------------------------------------------------------
    # Create bins from reference distribution
    # --------------------------------------------------------

    try:

        quantiles = np.linspace(
            0,
            1,
            bins + 1
        )

        breakpoints = np.unique(
            reference.quantile(
                quantiles
            ).values
        )

        if len(breakpoints) < 3:

            return 0.0

        reference_binned = pd.cut(
            reference,
            bins=breakpoints,
            include_lowest=True
        )

        current_binned = pd.cut(
            current,
            bins=breakpoints,
            include_lowest=True
        )

    except Exception:

        return np.nan

    # --------------------------------------------------------
    # Calculate distributions
    # --------------------------------------------------------

    reference_distribution = (
        reference_binned
        .value_counts(
            normalize=True,
            sort=False
        )
    )

    current_distribution = (
        current_binned
        .value_counts(
            normalize=True,
            sort=False
        )
    )

    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    epsilon = 0.0001

    reference_distribution = (
        reference_distribution
        .replace(0, epsilon)
    )

    current_distribution = (
        current_distribution
        .replace(0, epsilon)
    )

    # --------------------------------------------------------
    # PSI calculation
    #
    # PSI =
    # (Current % - Reference %)
    # *
    # ln(Current % / Reference %)
    # --------------------------------------------------------

    psi = (

        (
            current_distribution
            - reference_distribution
        )

        *

        np.log(
            current_distribution
            /
            reference_distribution
        )

    ).sum()

    return float(psi)


# ============================================================
# 9. INTERPRET PSI
# ============================================================

def interpret_psi(psi):

    if pd.isna(psi):

        return "UNAVAILABLE"

    if psi < 0.10:

        return "STABLE"

    elif psi < 0.25:

        return "MODERATE DRIFT"

    else:

        return "SIGNIFICANT DRIFT"


# ============================================================
# 10. CALCULATE DRIFT FOR ALL FEATURES
# ============================================================

print("\nCalculating feature drift...")

drift_results = []


for feature in FEATURE_COLUMNS:

    psi_value = calculate_psi(
        reference_df[feature],
        current_df[feature]
    )

    status = interpret_psi(
        psi_value
    )

    drift_results.append(
        {
            "Feature": feature,
            "PSI": psi_value,
            "Drift_Status": status
        }
    )


drift_df = pd.DataFrame(
    drift_results
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FEATURE DRIFT RESULTS")
print("=" * 60)

print(
    drift_df
    .sort_values(
        "PSI",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 12. OVERALL DRIFT
# ============================================================

significant_drift = int(
    (
        drift_df["Drift_Status"]
        == "SIGNIFICANT DRIFT"
    ).sum()
)

moderate_drift = int(
    (
        drift_df["Drift_Status"]
        == "MODERATE DRIFT"
    ).sum()
)

stable_features = int(
    (
        drift_df["Drift_Status"]
        == "STABLE"
    ).sum()
)


# ============================================================
# 13. OVERALL ADAPTIVE STATUS
# ============================================================

if significant_drift > 0:

    adaptive_status = (
        "ADAPTATION REQUIRED"
    )

elif moderate_drift > 0:

    adaptive_status = (
        "MONITOR CLOSELY"
    )

else:

    adaptive_status = (
        "STABLE"
    )


# ============================================================
# 14. CREATE SUMMARY
# ============================================================

summary = {

    "reference_observations":
        int(len(reference_df)),

    "current_observations":
        int(len(current_df)),

    "total_features":
        int(len(FEATURE_COLUMNS)),

    "stable_features":
        stable_features,

    "moderate_drift_features":
        moderate_drift,

    "significant_drift_features":
        significant_drift,

    "adaptive_status":
        adaptive_status
}


# ============================================================
# 15. SAVE DRIFT REPORT
# ============================================================

drift_df.to_csv(
    DRIFT_REPORT_FILE,
    index=False
)

print(
    f"\nSaved drift report: "
    f"{DRIFT_REPORT_FILE}"
)


# ============================================================
# 16. SAVE SUMMARY
# ============================================================

with open(
    DRIFT_SUMMARY_FILE,
    "w"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


print(
    f"Saved adaptive summary: "
    f"{DRIFT_SUMMARY_FILE}"
)


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("ADAPTIVE MONITORING SUMMARY")
print("=" * 60)

for key, value in summary.items():

    print(
        f"{key}: {value}"
    )


print("\nAdaptive monitoring complete.")