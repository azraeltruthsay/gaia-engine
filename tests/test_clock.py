"""The clock is right, and an answer that is the clock survives cleaning."""
from datetime import datetime, timezone
from unittest.mock import patch

from gaia_engine import config
from gaia_engine.core import _clean_response_text


def test_a_reply_that_is_only_the_clock_is_kept():
    assert _clean_response_text("[Clock: 2:46 PM PST (Local), Tuesday | 22:46 UTC]") == "2:46 PM PST (Local), Tuesday | 22:46 UTC"


def test_a_clock_trailing_a_reply_is_still_residue():
    assert _clean_response_text("Paris.\n[Clock: 2:46 PM PST]") == "Paris."


def test_the_clock_follows_daylight_saving(monkeypatch):
    monkeypatch.setenv("GAIA_USER_TZ", "America/Los_Angeles")
    with patch("datetime.datetime", wraps=datetime) as fake:
        fake.now.return_value = datetime(2026, 11, 3, 22, 46, tzinfo=timezone.utc)
        local, label = config.local_now()
        assert (local.hour, label) == (14, "PST")
        fake.now.return_value = datetime(2026, 7, 3, 22, 46, tzinfo=timezone.utc)
        local, label = config.local_now()
        assert (local.hour, label) == (15, "PDT")
