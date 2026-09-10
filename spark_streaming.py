# ============================================================
# SPARK STRUCTURED STREAMING
# FRAUD DETECTION PIPELINE
# ============================================================

import json
import os
import joblib
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    to_json,
    struct,
    lit
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType
)

import shap


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

INPUT_TOPIC = "transaction-events"

OUTPUT_TOPIC = "fraud-predictions"

MODEL_PATH = "rf_fraud_model.pkl"

CHECKPOINT_PATH = "./spark_checkpoint"


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("FraudDetectionStreaming")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("=" * 70)
print("SPARK FRAUD DETECTION STREAMING")
print("=" * 70)

print("Input topic :", INPUT_TOPIC)
print("Output topic:", OUTPUT_TOPIC)


# ============================================================
# KAFKA MESSAGE SCHEMA
# ============================================================

schema_fields = [
    StructField("transaction_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("Time", DoubleType(), True),
]

for feature in [f"V{i}" for i in range(1, 29)]:
    schema_fields.append(
        StructField(feature, DoubleType(), True)
    )

schema_fields.append(
    StructField("Amount", DoubleType(), True)
)

schema_fields.append(
    StructField("actual_label", DoubleType(), True)
)

schema = StructType(schema_fields)


# ============================================================
# READ KAFKA STREAM
# ============================================================

raw_stream = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        KAFKA_BOOTSTRAP_SERVERS
    )
    .option(
        "subscribe",
        INPUT_TOPIC
    )
    .option(
        "startingOffsets",
        "latest"
    )
    .load()
)


# ============================================================
# CONVERT KAFKA VALUE FROM BINARY TO JSON
# ============================================================

json_stream = (
    raw_stream
    .selectExpr(
        "CAST(value AS STRING) AS json_value"
    )
)


# ============================================================
# PARSE JSON
# ============================================================

parsed_stream = (
    json_stream
    .select(
        from_json(
            col("json_value"),
            schema
        ).alias("data")
    )
    .select("data.*")
)


# ============================================================
# VALIDATE REQUIRED FEATURES
# ============================================================

required_columns = [
    "transaction_id"
] + FEATURE_COLUMNS


clean_stream = (
    parsed_stream
    .dropna(
        subset=FEATURE_COLUMNS
    )
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

print("Random Forest model loaded.")


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model
)


# ============================================================
# PROCESS EACH MICRO-BATCH
# ============================================================

def process_batch(
    batch_df,
    batch_id
):

    if batch_df.rdd.isEmpty():

        print(
            f"Batch {batch_id}: no records."
        )

        return

    print()
    print("=" * 70)
    print(
        f"Processing Spark micro-batch: {batch_id}"
    )
    print(
        f"Records: {batch_df.count()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # CONVERT TO PANDAS
    # --------------------------------------------------------

    pandas_df = batch_df.toPandas()


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    X = pandas_df[
        FEATURE_COLUMNS
    ].copy()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        X
    )


    # --------------------------------------------------------
    # PROCESS EACH TRANSACTION
    # --------------------------------------------------------

    output_records = []


    for i in range(
        len(pandas_df)
    ):

        transaction = pandas_df.iloc[i]

        prediction = int(
            predictions[i]
        )

        probability = float(
            probabilities[i]
        )


        # ----------------------------------------------------
        # EXTRACT FRAUD SHAP VALUES
        # ----------------------------------------------------

        if isinstance(
            shap_values,
            list
        ):

            fraud_shap = np.asarray(
                shap_values[1]
            )[i]

        else:

            shap_array = np.asarray(
                shap_values
            )

            if shap_array.ndim == 3:

                fraud_shap = (
                    shap_array[
                        i,
                        :,
                        1
                    ]
                )

            elif shap_array.ndim == 2:

                fraud_shap = (
                    shap_array[i]
                )

            else:

                fraud_shap = shap_array


        # ----------------------------------------------------
        # TOP SHAP FEATURES
        # ----------------------------------------------------

        shap_table = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "SHAP_Value": fraud_shap
            }
        )

        shap_table[
            "Absolute_SHAP"
        ] = shap_table[
            "SHAP_Value"
        ].abs()

        shap_table = (
            shap_table
            .sort_values(
                "Absolute_SHAP",
                ascending=False
            )
            .head(10)
        )


        top_shap_features = []


        for _, row in shap_table.iterrows():

            top_shap_features.append(
                {
                    "Feature":
                        str(row["Feature"]),

                    "SHAP_Value":
                        float(row["SHAP_Value"]),

                    "Direction":
                        (
                            "Pushes toward Fraud"
                            if row["SHAP_Value"] >= 0
                            else
                            "Pushes toward Legitimate"
                        )
                }
            )


        # ----------------------------------------------------
        # CREATE OUTPUT RECORD
        # ----------------------------------------------------

        result = {

            "transaction_id":
                str(
                    transaction.get(
                        "transaction_id",
                        ""
                    )
                ),

            "timestamp":
                str(
                    transaction.get(
                        "timestamp",
                        ""
                    )
                ),

            "prediction":
                (
                    "FRAUD"
                    if prediction == 1
                    else "LEGITIMATE"
                ),

            "fraud_probability":
                probability,

            "top_shap_features":
                top_shap_features
        }


        # ----------------------------------------------------
        # INCLUDE ORIGINAL FEATURES
        # ----------------------------------------------------

        for feature in FEATURE_COLUMNS:

            try:

                result[feature] = float(
                    transaction[feature]
                )

            except Exception:

                result[feature] = 0.0


        # ----------------------------------------------------
        # ACTUAL LABEL IF AVAILABLE
        # ----------------------------------------------------

        if "actual_label" in pandas_df.columns:

            actual_value = transaction.get(
                "actual_label"
            )

            if pd.notna(
                actual_value
            ):

                result[
                    "actual_label"
                ] = int(
                    actual_value
                )


        output_records.append(
            result
        )


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            f"Transaction: {result['transaction_id']}"
        )

        print(
            f"Prediction: {result['prediction']}"
        )

        print(
            f"Fraud probability: "
            f"{probability:.2%}"
        )


    # ========================================================
    # SEND RESULTS TO KAFKA
    # ========================================================

    output_json = pd.DataFrame(
        {
            "value": [
                json.dumps(
                    record
                )
                for record in output_records
            ]
        }
    )


    output_spark_df = (
        spark.createDataFrame(
            output_json
        )
    )


    (
        output_spark_df
        .selectExpr(
            "CAST(value AS STRING) AS value"
        )
        .write
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            KAFKA_BOOTSTRAP_SERVERS
        )
        .option(
            "topic",
            OUTPUT_TOPIC
        )
        .save()
    )


    print(
        f"Batch {batch_id} successfully sent "
        f"to Kafka topic '{OUTPUT_TOPIC}'."
    )


# ============================================================
# START STREAMING QUERY
# ============================================================

query = (
    clean_stream
    .writeStream
    .foreachBatch(
        process_batch
    )
    .option(
        "checkpointLocation",
        CHECKPOINT_PATH
    )
    .trigger(
        processingTime="2 seconds"
    )
    .start()
)


print()
print("=" * 70)
print("SPARK STREAMING IS RUNNING")
print("=" * 70)
print(
    "Waiting for transactions..."
)
print()


query.awaitTermination()