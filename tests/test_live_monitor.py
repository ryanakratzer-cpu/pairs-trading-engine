"""Tests for run_live_monitor pure helpers: spread/z math, band classification,
HTML rendering, and stream fallback decisions. No network, no websockets."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import run_live_monitor as rlm
from signals.spread import SignalConfig


@pytest.fixture
def config():
    return SignalConfig(zscore_window=50, entry_z=2.0, exit_z=0.5, stop_z=3.0, max_holding_bars=62)


@pytest.fixture
def context():
    """Synthetic daily context with hand-pickable numbers."""
    dates = pd.bdate_range("2024-01-01", periods=90, tz="UTC")
    return {
        "hedge_ratio": 0.8,
        "intercept": 0.1,
        "adf_pvalue": 0.012,
        "half_life_days": 9.5,
        "spread_mean": 0.02,
        "spread_std": 0.01,
        "kalman_beta": 0.79,
        "daily_spread_tail": pd.Series(np.zeros(90), index=dates),
        "last_daily_close": {"AAA": 100.0, "BBB": 50.0},
    }


@pytest.fixture
def history():
    idx = pd.DatetimeIndex(pd.date_range("2024-06-03 14:30", periods=3, freq="1min", tz="UTC"))
    return pd.DataFrame(
        {
            "price_a": [100.0, 100.5, 101.0],
            "price_b": [50.0, 50.1, 50.2],
            "spread": [0.02, 0.021, 0.022],
            "zscore": [0.0, 0.1, 0.2],
            "age_minutes": [0.5, 0.5, 0.5],
            "is_stale": [False, False, False],
        },
        index=idx,
    )


# ---------------------------------------------------------------- compute math


def test_compute_spread_point_hand_computed(context):
    spread, z = rlm.compute_spread_point(100.0, 50.0, context)
    expected_spread = math.log(100.0) - 0.8 * math.log(50.0) - 0.1
    assert spread == pytest.approx(expected_spread, abs=1e-12)
    assert z == pytest.approx((expected_spread - 0.02) / 0.01, abs=1e-9)


def test_compute_spread_point_zero_z_at_mean(context):
    # Choose price_a so the spread lands exactly on spread_mean => z == 0.
    price_b = 50.0
    price_a = math.exp(context["spread_mean"] + context["intercept"] + context["hedge_ratio"] * math.log(price_b))
    spread, z = rlm.compute_spread_point(price_a, price_b, context)
    assert spread == pytest.approx(context["spread_mean"], abs=1e-12)
    assert z == pytest.approx(0.0, abs=1e-9)


def test_compute_spread_point_matches_poll_loop_formula(context):
    # Same formula the old inline poll-loop code used.
    pa, pb = 37.5, 221.4
    spread, z = rlm.compute_spread_point(pa, pb, context)
    old_spread = np.log(pa) - context["hedge_ratio"] * np.log(pb) - context["intercept"]
    old_z = (old_spread - context["spread_mean"]) / context["spread_std"]
    assert spread == pytest.approx(old_spread)
    assert z == pytest.approx(old_z)


# ------------------------------------------------------------- classify bands


@pytest.mark.parametrize(
    ("z", "expected_fragment"),
    [
        (3.0, "BEYOND STOP BAND"),  # stop edge inclusive
        (-3.0, "BEYOND STOP BAND"),
        (3.5, "BEYOND STOP BAND"),
        (2.0, "SHORT-SPREAD ENTRY ZONE"),  # entry edge inclusive
        (2.99, "SHORT-SPREAD ENTRY ZONE"),
        (-2.0, "LONG-SPREAD ENTRY ZONE"),
        (-2.99, "LONG-SPREAD ENTRY ZONE"),
        (0.5, "MEAN ZONE"),  # exit edge inclusive
        (-0.5, "MEAN ZONE"),
        (0.0, "MEAN ZONE"),
        (0.51, "NEUTRAL"),
        (1.99, "NEUTRAL"),
        (-1.99, "NEUTRAL"),
    ],
)
def test_classify_band_edges(config, z, expected_fragment):
    assert expected_fragment in rlm.classify(z, config)


# ---------------------------------------------------------------- render_html


def test_render_html_stream_badge_and_disclaimer(tmp_path, monkeypatch, context, history, config):
    monkeypatch.setattr(rlm, "OUTPUTS_DIR", tmp_path)
    rlm.render_html("AAA", "BBB", context, history, config, mode="stream")
    html = (tmp_path / "live_monitor.html").read_text(encoding="utf-8")
    assert "STREAMING (Yahoo websocket)" in html
    assert rlm.DISCLAIMER in html
    assert 'content="5"' in html  # 5s meta refresh in stream mode


def test_render_html_poll_badge_and_refresh(tmp_path, monkeypatch, context, history, config):
    monkeypatch.setattr(rlm, "OUTPUTS_DIR", tmp_path)
    rlm.render_html("AAA", "BBB", context, history, config, mode="poll")
    html = (tmp_path / "live_monitor.html").read_text(encoding="utf-8")
    assert "POLLING" in html
    assert "STREAMING (Yahoo websocket)" not in html
    assert rlm.DISCLAIMER in html
    assert 'content="15"' in html  # 15s meta refresh in poll mode


def test_render_html_stale_note(tmp_path, monkeypatch, context, history, config):
    monkeypatch.setattr(rlm, "OUTPUTS_DIR", tmp_path)
    stale = history.copy()
    stale["is_stale"] = True
    rlm.render_html("AAA", "BBB", context, stale, config, mode="stream")
    html = (tmp_path / "live_monitor.html").read_text(encoding="utf-8")
    assert "Market closed / data stale" in html


# ------------------------------------------------------- stream fallback logic


def test_stream_action_waits_while_connecting():
    assert rlm.decide_stream_action(False, True, 5.0) == "wait"


def test_stream_action_fallback_on_connect_timeout():
    assert rlm.decide_stream_action(False, True, 15.0) == "fallback"
    assert rlm.decide_stream_action(False, True, 30.0) == "fallback"


def test_stream_action_fallback_when_worker_dies_before_connecting():
    assert rlm.decide_stream_action(False, False, 1.0) == "fallback"


def test_stream_action_fallback_on_disconnect_after_connecting():
    assert rlm.decide_stream_action(True, False, 1.0) == "fallback"


def test_stream_action_quiet_when_no_ticks():
    assert rlm.decide_stream_action(True, True, 60.0) == "quiet"
    assert rlm.decide_stream_action(True, True, 3600.0) == "quiet"


def test_stream_action_ok_when_ticking():
    assert rlm.decide_stream_action(True, True, 0.5) == "ok"
    assert rlm.decide_stream_action(True, True, 59.9) == "ok"
