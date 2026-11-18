"""A client that hangs up before the response is sent is not an engine
error: no traceback, one log line (GAIA ml4a)."""
from gaia_engine.manager import ManagedEngineHandler


class _Gone:
    def write(self, b):
        raise BrokenPipeError(32, "Broken pipe")


def test_broken_pipe_is_swallowed(caplog):
    h = ManagedEngineHandler.__new__(ManagedEngineHandler)
    h.command, h.path, h.request_version = "GET", "/health", "HTTP/1.1"
    h._headers_buffer = []
    h.wfile = _Gone()
    h.send_response = lambda status: None
    h.send_header = lambda k, v: None
    with caplog.at_level("INFO"):
        h._send_proxy_response(200, {}, b"{}")
    assert "Client disconnected" in caplog.text
