# ============================================================
# XAI DSS
# SYNTHETIC TRANSACTION GENERATOR
# + KAFKA PRODUCER
# + KAFKA CONSUMER
# + FRAUD DETECTION MODEL
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
# 4. SYNTHETIC TRANSACTION GENERATOR
# ============================================================

def generate_transaction():

    # Generate V1-V28 jointly
    v_values = np.random.multivariate_normal(
        mean=v_mean,
        cov=v_cov
    )

    # Generate Time
    transaction_time = np.random.uniform(
        time_min,
        time_max
    )

    # Generate Amount
    transaction_amount = np.random.lognormal(
        mean=np.log(amount_median + 1),
        sigma=1.0
    )

    # Build transaction
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

    # Return exactly the 30 features
    # expected by the trained model
    return pd.DataFrame(
        [transaction],
        columns=feature_columns
    )


# ============================================================
# 5. KAFKA PRODUCER
# ============================================================

producer = KafkaProducer(

    bootstrap_servers="localhost:9092",

    value_serializer=lambda value:
        json.dumps(value).encode("utf-8")
)

print("Kafka producer connected.")


# ============================================================
# 6. PRODUCER FUNCTION
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

            # Add transaction ID
            transaction_data["transaction_id"] = (
                str(uuid.uuid4())
            )

            # Send transaction to Kafka
            producer.send(
                "transaction-events",
                transaction_data
            )

            producer.flush()

            print("\nTransaction sent to Kafka:")
            print(transaction_data)

            # Wait 2 seconds
            time.sleep(2)

    except KeyboardInterrupt:

        print("\nProducer stopped.")

    finally:

        producer.close()


# ============================================================
# 7. KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(

    "transaction-events",

    bootstrap_servers="localhost:9092",

    group_id="fraud-detection-model",

    auto_offset_reset="latest",

    value_deserializer=lambda value:
        json.loads(
            value.decode("utf-8")
        )
)

print("Kafka consumer connected.")

print("Waiting for transactions...")


# ============================================================
# 8. CONSUMER + FRAUD DETECTION
# ============================================================

def consume_transactions():

    for message in consumer:

        # Get transaction from Kafka
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
        # Extract ONLY the 30 features used by the model
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
        # Fraud prediction
        # ----------------------------------------------------

        prediction = rf_model.predict(
            model_input
        )[0]

        # ----------------------------------------------------
        # Fraud probability
        # ----------------------------------------------------

        probability = rf_model.predict_proba(
            model_input
        )[0][1]

        # ----------------------------------------------------
        # Interpret prediction
        # ----------------------------------------------------

        if prediction == 1:

            result = "FRAUD"

        else:

            result = "LEGITIMATE"

        print(
            "Prediction:",
            result
        )

        print(
            f"Fraud probability: "
            f"{probability:.2%}"
        )


# ============================================================
# 9. START PRODUCER IN BACKGROUND
# ============================================================

producer_thread = Thread(
    target=produce_transactions,
    daemon=True
)

producer_thread.start()


# ============================================================
# 10. START CONSUMER
# ============================================================

consume_transactions()

import json
import time
import uuid
import joblib
import numpy as np
import pandas as pd
import shap

from kafka import KafkaProducer, KafkaConsumer
from threading import Thread