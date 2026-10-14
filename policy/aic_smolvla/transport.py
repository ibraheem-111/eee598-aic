"""Local request/response link between the ROS policy and the inference server.

The AIC pixi env's PyTorch (2.7, CUDA 12.6) can't run on the RTX 5070 (sm_120) and lacks SmolVLA's
dependencies, so SmolVLA runs in the LeRobot venv and the two talk over a localhost socket.
Messages are plain dicts of numpy arrays / str / numbers (pickled by multiprocessing.connection).
"""

import os
from multiprocessing.connection import Client, Listener

DEFAULT_ADDRESS = ("127.0.0.1", int(os.environ.get("AIC_SMOLVLA_PORT", "50765")))
AUTHKEY = os.environ.get("AIC_SMOLVLA_AUTHKEY", "eee598-aic").encode()


def listen(address=DEFAULT_ADDRESS) -> Listener:
    return Listener(address, authkey=AUTHKEY)


def connect(address=DEFAULT_ADDRESS):
    return Client(address, authkey=AUTHKEY)


def request(conn, msg: dict) -> dict:
    conn.send(msg)
    reply = conn.recv()
    if reply.get("error"):
        raise RuntimeError(f"SmolVLA server error: {reply['error']}")
    return reply
