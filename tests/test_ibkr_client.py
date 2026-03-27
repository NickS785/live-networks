from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from live_cta.sources.ibkr_client import IBKRConfig, IBKRContract, IBKRTickDataSource


def _make_source(
    *,
    base_url: str = "https://localhost:5000",
    account_id: str = "",
    verify_ssl: bool = False,
    tz: str = "America/Chicago",
    conid: str = "123456",
    ticker: str = "NG",
    bar_size: str = "1min",
) -> IBKRTickDataSource:
    return IBKRTickDataSource(
        config=IBKRConfig(
            base_url=base_url,
            account_id=account_id,
            verify_ssl=verify_ssl,
            timeout=10.0,
        ),
        contracts={ticker: IBKRContract(conid=conid, ticker=ticker, bar_size=bar_size)},
        tz=tz,
    )


def _history_payload(start_utc: str = "2026-03-20 14:00:00+00:00", periods: int = 3) -> dict:
    start = pd.Timestamp(start_utc)
    bars = []
    for idx in range(periods):
        ts = start + pd.Timedelta(minutes=idx)
        bars.append(
            {
                "t": int(ts.value // 1_000_000),
                "o": 4.0 + idx * 0.1,
                "h": 4.1 + idx * 0.1,
                "l": 3.9 + idx * 0.1,
                "c": 4.05 + idx * 0.1,
                "v": 10 + idx,
            }
        )
    return {"data": bars}


def test_ibkr_get_history_formats_history_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_source()
    captured: Dict[str, Any] = {}
    payload = _history_payload(periods=2)

    def fake_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        captured["path"] = path
        captured["params"] = params
        return payload

    monkeypatch.setattr(source, "_get", fake_get)

    frame = source.get_history(conid="123456", period="2d", bar_size="1min", outside_rth=True)

    assert captured["path"] == "/iserver/marketdata/history"
    assert captured["params"] == {
        "conid": "123456",
        "period": "2d",
        "bar": "1min",
        "outsideRth": "true",
    }
    assert list(frame.columns) == ["Datetime", "Open", "High", "Low", "Close", "TotalVolume"]
    assert len(frame) == 2
    assert frame["Datetime"].iloc[0] == pd.Timestamp("2026-03-20 14:00:00", tz="UTC")
    assert frame["Close"].tolist() == pytest.approx([4.05, 4.15])
    assert frame["TotalVolume"].tolist() == [10, 11]


def test_ibkr_get_ticks_treats_epoch_millis_as_utc_before_localizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source()
    payload = _history_payload(periods=2)

    def fake_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        assert path == "/iserver/marketdata/history"
        assert params is not None
        assert params["conid"] == "123456"
        assert params["bar"] == "1min"
        return payload

    monkeypatch.setattr(source, "_get", fake_get)
    monkeypatch.setattr(source, "ensure_alive", lambda: None)

    ticks = source.get_ticks(
        "NG",
        pd.Timestamp("2026-03-20 09:00:00", tz="America/Chicago"),
        pd.Timestamp("2026-03-20 09:01:00", tz="America/Chicago"),
    )

    expected_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-03-20 09:00:00", tz="America/Chicago"),
            pd.Timestamp("2026-03-20 09:01:00", tz="America/Chicago"),
        ]
    )
    assert ticks.index.equals(expected_index)
    assert ticks["Close"].tolist() == pytest.approx([4.05, 4.15])
    assert ticks["TotalVolume"].tolist() == [10, 11]
    assert ticks["BidVolume"].tolist() == [5.0, 5.5]
    assert ticks["AskVolume"].tolist() == [5.0, 5.5]
    assert ticks["NumTrades"].tolist() == [1, 1]


def test_ibkr_live_gateway_smoke() -> None:
    gateway_url = os.getenv("CTAFLOW_IBKR_GATEWAY_URL")
    conid = os.getenv("CTAFLOW_IBKR_CONID")
    if not gateway_url or not conid:
        pytest.skip("Set CTAFLOW_IBKR_GATEWAY_URL and CTAFLOW_IBKR_CONID to run the live IBKR smoke test.")

    ticker = os.getenv("CTAFLOW_IBKR_TICKER", "NG")
    account_id = os.getenv("CTAFLOW_IBKR_ACCOUNT_ID", "")
    verify_ssl = os.getenv("CTAFLOW_IBKR_VERIFY_SSL", "").lower() in {"1", "true", "yes"}
    period = os.getenv("CTAFLOW_IBKR_PERIOD", "2d")
    bar_size = os.getenv("CTAFLOW_IBKR_BAR_SIZE", "1min")
    tz = os.getenv("CTAFLOW_IBKR_TZ", "America/Chicago")

    source = _make_source(
        base_url=gateway_url,
        account_id=account_id,
        verify_ssl=verify_ssl,
        tz=tz,
        conid=conid,
        ticker=ticker,
        bar_size=bar_size,
    )

    keepalive = source.tickle()
    assert isinstance(keepalive, dict)

    history = source.get_history(conid=conid, period=period, bar_size=bar_size, outside_rth=True)
    assert not history.empty
    assert {"Datetime", "Open", "High", "Low", "Close", "TotalVolume"}.issubset(history.columns)

    localized = history.copy()
    localized["Datetime"] = pd.to_datetime(localized["Datetime"], utc=True).dt.tz_convert(tz)
    localized = localized.set_index("Datetime").sort_index()

    end_time = localized.index.max()
    start_time = end_time - pd.Timedelta(minutes=120)
    ticks = source.get_ticks(ticker, start_time, end_time)

    assert not ticks.empty
    assert ticks.index.tz is not None
    assert {"Close", "TotalVolume", "BidVolume", "AskVolume", "NumTrades"}.issubset(ticks.columns)
    assert (ticks["TotalVolume"] >= 0).all()
