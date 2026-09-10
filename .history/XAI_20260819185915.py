# ==============================
# 1. KAFKA CONNECTION
# ==============================

from kafka import KafkaProducer
import json

# Connect to Kafka
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)

print("Kafka connection successful!")

# ==============================
# 2. TEST TRANSACTION
# ==============================

transaction = {
    "transaction_id": "TX002",
    "amount": 2500,
    "merchant": "ONLINE_STORE",
    "country": "NG"
}


# ==============================
# 3. SEND TRANSACTION TO KAFKA
# ==============================

producer.send("transaction-events", transaction)

producer.flush()

print("Transaction sent successfully!")
print(transaction)

producer.close()

import joblib

synthetic_parameter = joblib.load("synthetic_parameter.pkl")

print("Synthetic parameters loaded successfully.")

# ============================================================
# 1. LOAD SYNTHETIC PARAMETERS
# ============================================================

import joblib
import numpy as np
import pandas as pd

synthetic_parameter = joblib.load("synthetic_parameter.pkl")

feature_columns = synthetic_parameter["feature_columns"]
v_cols = synthetic_parameter["v_cols"]

v_mean = synthetic_parameter["v_mean"]
v_cov = synthetic_parameter["v_cov"]

time_min = synthetic_parameter["time_min"]
time_max = synthetic_parameter["time_max"]

amount_median = synthetic_parameter["amount_median"]

print("Synthetic parameters loaded successfully.")


# ============================================================
# 2. SYNTHETIC TRANSACTION GENERATOR
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
            for col, value in zip(v_cols, v_values)
        }
    )

    # Return the 30 features expected by the model
    return pd.DataFrame(
        [transaction],
        columns=feature_columns
    )


# ============================================================
# 3. TEST THE GENERATOR
# ============================================================

synthetic_transaction = generate_transaction()

print("\nSynthetic transaction generated:")
print(synthetic_transaction)

print("\nNumber of features:", synthetic_transaction.shape[1])


# ============================================================
# 4. KAFKA CONNECTION
# ============================================================

import json
import time
import uuid

from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)

print("\nKafka connection successful!")


# ============================================================
# 5. GENERATE AND SEND TRANSACTIONS TO KAFKA
# ============================================================

try:

    while True:

        # Generate synthetic transaction
        synthetic_transaction = generate_transaction()

        # Convert DataFrame row to dictionary
        transaction_data = synthetic_transaction.iloc[0].to_dict()

        # Add unique ID for tracking
        transaction_data["transaction_id"] = str(uuid.uuid4())

        # Send transaction to Kafka
        producer.send(
            "transaction-events",
            transaction_data
        )

        producer.flush()

        print("\nTransaction sent to Kafka:")
        print(transaction_data)

        # Generate a transaction every 2 seconds
        time.sleep(2)


except KeyboardInterrupt:

    print("\nTransaction generator stopped.")


finally:

    producer.close()

    print("Kafka connection closed.")
    
