"""Run with PYTHONPATH=scripts pytest -p offline_test_guard to forbid inference."""

import socket

import pytest


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    original = socket.socket.connect

    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "localhost", "::1"):
            raise AssertionError("External network forbidden in offline regression tests")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    import litellm

    def forbidden(*args, **kwargs):
        raise AssertionError("Real LLM calls forbidden in offline regression tests")

    async def aforbidden(*args, **kwargs):
        raise AssertionError("Real LLM calls forbidden in offline regression tests")

    monkeypatch.setattr(litellm, "completion", forbidden)
    monkeypatch.setattr(litellm, "acompletion", aforbidden)
