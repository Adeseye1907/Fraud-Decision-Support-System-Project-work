# ============================================================
# WEBSOCKET SERVER
# FRAUD DSS REAL-TIME DELIVERY LAYER
# ============================================================

import asyncio
import json

from kafka import KafkaConsumer

import websockets


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_SERVER = "localhost:9092"

KAFKA_TOPIC = "fraud-predictions"

WEBSOCKET_HOST = "localhost"

WEBSOCKET_PORT = 8765


# ============================================================
# CONNECTED CLIENTS
# ============================================================

connected_clients = set()


# ============================================================
# KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(
    KAFKA_TOPIC,

    bootstrap_servers=KAFKA_SERVER,

    value_deserializer=lambda message:
        json.loads(
            message.decode("utf-8")
        ),

    auto_offset_reset="latest",

    enable_auto_commit=True,

    group_id="fraud-websocket-server"
)


# ============================================================
# REGISTER CLIENT
# ============================================================

async def websocket_handler(
    websocket
):

    connected_clients.add(
        websocket
    )

    print(
        "WebSocket client connected."
    )

    try:

        await websocket.wait_closed()

    finally:

        connected_clients.discard(
            websocket
        )

        print(
            "WebSocket client disconnected."
        )


# ============================================================
# BROADCAST MESSAGE
# ============================================================

async def broadcast_message(
    message
):

    if not connected_clients:

        return

    disconnected = set()

    for client in connected_clients:

        try:

            await client.send(
                json.dumps(
                    message
                )
            )

        except Exception:

            disconnected.add(
                client
            )


    for client in disconnected:

        connected_clients.discard(
            client
        )


# ============================================================
# KAFKA READER
# ============================================================

async def kafka_reader():

    print(
        "WebSocket Kafka reader started."
    )

    while True:

        messages = consumer.poll(
            timeout_ms=500,
            max_records=20
        )

        for _, records in messages.items():

            for message in records:

                transaction = (
                    message.value
                )

                print(
                    "\nWebSocket broadcasting:"
                )

                print(
                    transaction
                )

                await broadcast_message(
                    transaction
                )

        await asyncio.sleep(
            0.1
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 60)

    print(
        "FRAUD DSS WEBSOCKET SERVER"
    )

    print("=" * 60)

    print(
        f"WebSocket: "
        f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}"
    )

    print(
        f"Kafka topic: {KAFKA_TOPIC}"
    )


    async with websockets.serve(
        websocket_handler,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT
    ):

        await kafka_reader()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )