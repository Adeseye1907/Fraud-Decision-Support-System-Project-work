# ============================================================
# XAI DSS
# SYNTHETIC TRANSACTION GENERATOR
# + KAFKA PRODUCER
# + KAFKA CONSUMER
# + RANDOM FOREST FRAUD DETECTION
# + SHAP EXPLAINABILITY
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import json
import time
import uuid
import joblib

import numpy as np
import pandas as pd
import shap

from kafka import KafkaProducer, KafkaConsumer
from threading import Thread


# ============================================================
# 2. LOAD SYNTHETIC GENERATOR PARAMETERS
# ============================================================

synthetic_parameter = joblib.load(
    "synthetic_parameter.pkl"
)

feature_columns = synthetic_parameter["feature_columns"]

v_cols = synthetic_parameter["v_cols"]

v_mean = synthetic_parameter["v_mean"]

v_cov = synthetic_parameter["v_cov"]

time_min = synthetic_parameter["time_min"]

time_max = synthetic_parameter["time_max"]

amount_median = synthetic_parameter["amount_median"]

print("Synthetic parameters loaded successfully.")


# ============================================================
# 3. LOAD TRAINED FRAUD DETECTION MODEL
# ============================================================

rf_model = joblib.load(
    "rf_fraud_model.pkl"
)

print("Fraud detection model loaded successfully.")


# ============================================================
# 4. LOAD SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    rf_model
)

print("SHAP explainer loaded successfully.")


# ============================================================
# 5. SYNTHETIC TRANSACTION GENERATOR
# ============================================================

def generate_transaction():

    # --------------------------------------------------------
    # Generate V1-V28 jointly
    # --------------------------------------------------------

    v_values = np.random.multivariate_normal(
        mean=v_mean,
        cov=v_cov
    )

    # --------------------------------------------------------
    # Generate Time
    # --------------------------------------------------------

    transaction_time = np.random.uniform(
        time_min,
        time_max
    )

    # --------------------------------------------------------
    # Generate Amount
    # --------------------------------------------------------

    transaction_amount = np.random.lognormal(
        mean=np.log(
            amount_median + 1
        ),
        sigma=1.0
    )

    # --------------------------------------------------------
    # Build transaction
    # --------------------------------------------------------

    transaction = {
        "Time": transaction_time,
        "Amount": transaction_amount
    }

    transaction.update(
        {
            col: value
            for col, value in zip(
                v_cols,
                v_values
            )
        }
    )

    # --------------------------------------------------------
    # Return exactly the 30 features expected
    # by the trained Random Forest
    # --------------------------------------------------------

    return pd.DataFrame(
        [transaction],
        columns=feature_columns
    )


# ============================================================
# 6. KAFKA PRODUCER
# ============================================================

producer = KafkaProducer(

    bootstrap_servers="localhost:9092",

    value_serializer=lambda value:
        json.dumps(value).encode("utf-8")
)

print("Kafka producer connected.")


# ============================================================
# 7. PRODUCER FUNCTION
# ============================================================

def produce_transactions():

    try:

        while True:

            # Generate synthetic transaction
            synthetic_transaction = (
                generate_transaction()
            )

            # Convert DataFrame row to dictionary
            transaction_data = (
                synthetic_transaction
                .iloc[0]
                .to_dict()
            )

            # Add unique transaction ID
            transaction_data["transaction_id"] = (
                str(uuid.uuid4())
            )

            # Send transaction to Kafka
            producer.send(
                "transaction-events",
                transaction_data
            )

            producer.flush()

            print("\n")
            print("=" * 60)
            print("TRANSACTION SENT TO KAFKA")
            print("=" * 60)

            print(transaction_data)

            # Wait 2 seconds
            time.sleep(2)

    except KeyboardInterrupt:

        print(
            "\nProducer stopped."
        )

    finally:

        producer.close()

        print(
            "Kafka producer closed."
        )


# ============================================================
# 8. KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(

    "transaction-events",

    bootstrap_servers="localhost:9092",

    group_id="fraud-detection-xai-v1",

    auto_offset_reset="latest",

    enable_auto_commit=True,

    value_deserializer=lambda value:
        json.loads(
            value.decode("utf-8")
        )
)

print("Kafka consumer connected.")

print("Waiting for transactions...")


# ============================================================
# 9. CONSUMER + FRAUD DETECTION + SHAP
# ============================================================

def consume_transactions():

    for message in consumer:

        # ----------------------------------------------------
        # Receive transaction from Kafka
        # ----------------------------------------------------

        transaction_data = message.value

        print("\n")
        print("=" * 60)
        print("TRANSACTION RECEIVED")
        print("=" * 60)

        print(
            "Transaction ID:",
            transaction_data["transaction_id"]
        )

        # ----------------------------------------------------
        # Extract ONLY the 30 model features
        # ----------------------------------------------------

        model_input = pd.DataFrame(
            [
                {
                    column:
                    transaction_data[column]

                    for column in feature_columns
                }
            ]
        )

        # ----------------------------------------------------
        # Random Forest prediction
        # ----------------------------------------------------

        prediction = rf_model.predict(
            model_input
        )[0]

        # ----------------------------------------------------
        # Fraud probability
        # ----------------------------------------------------

        probability = (
            rf_model
            .predict_proba(
                model_input
            )[0][1]
        )

        # ----------------------------------------------------
        # Interpret prediction
        # ----------------------------------------------------

        if prediction == 1:

            result = "FRAUD"

        else:

            result = "LEGITIMATE"

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        print("\n--- FRAUD DETECTION RESULT ---")

        print(
            "Prediction:",
            result
        )

        print(
            f"Fraud probability: "
            f"{probability:.2%}"
        )

        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        try:

            # Calculate SHAP values
            shap_values = (
                explainer.shap_values(
                    model_input
                )
            )

            # ------------------------------------------------
            # Handle different SHAP output formats
            # ------------------------------------------------

            if isinstance(
                shap_values,
                list
            ):

                # Class 1 = FRAUD
                fraud_shap_values = (
                    shap_values[1][0]
                )

            else:

                shap_values = np.asarray(
                    shap_values
                )

                if shap_values.ndim == 3:

                    fraud_shap_values = (
                        shap_values[
                            0,
                            :,
                            1
                        ]
                    )

                else:

                    fraud_shap_values = (
                        shap_values[0]
                    )

            # ------------------------------------------------
            # Create readable SHAP DataFrame
            # ------------------------------------------------

            shap_df = pd.DataFrame({

                "Feature":
                    feature_columns,

                "Feature_Value":
                    model_input.iloc[0].values,

                "SHAP_Value":
                    fraud_shap_values
            })

            # ------------------------------------------------
            # Calculate absolute contribution
            # ------------------------------------------------

            shap_df["Absolute_SHAP"] = (
                shap_df[
                    "SHAP_Value"
                ].abs()
            )

            # ------------------------------------------------
            # Sort strongest contributors first
            # ------------------------------------------------

            shap_df = shap_df.sort_values(
                "Absolute_SHAP",
                ascending=False
            )

            # ------------------------------------------------
            # Display top 5 features
            # ------------------------------------------------

            print(
                "\n--- TOP 5 SHAP FEATURES ---"
            )

            print(
                shap_df[
                    [
                        "Feature",
                        "Feature_Value",
                        "SHAP_Value"
                    ]
                ]
                .head(5)
                .to_string(
                    index=False
                )
            )

        except Exception as e:

            print(
                "\nSHAP explanation error:"
            )

            print(e)

 # ----------------------------------------------------
        # Publish enriched result for the dashboard
        # ----------------------------------------------------
DASHBOARD_TOPIC = "fraud-predictions"
enriched_transaction = dict(transaction_data)

        enriched_transaction["prediction"] = result
        enriched_transaction["fraud_probability"] = float(probability)
        enriched_transaction["timestamp"] = pd.Timestamp.utcnow().isoformat()

        try:
            enriched_transaction["top_shap_features"] = (
                shap_df[["Feature", "Feature_Value", "SHAP_Value"]]
                .head(5)
                .to_dict(orient="records")
            )
        except NameError:
            enriched_transaction["top_shap_features"] = []

        producer.send(DASHBOARD_TOPIC, enriched_transaction)
        producer.flush()
        
# ============================================================
# 10. START PRODUCER IN BACKGROUND
# ============================================================

producer_thread = Thread(
    target=produce_transactions,
    daemon=True
)

producer_thread.start()


# ============================================================
# 11. START CONSUMER
# ============================================================

consume_transactions()