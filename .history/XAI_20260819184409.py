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