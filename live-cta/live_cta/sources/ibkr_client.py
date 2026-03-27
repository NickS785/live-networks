"""IBKR Web API (Client Portal) tick data source for CTAFlow live pipelines.

Assumes the IBKR Client Portal API Gateway is running and authenticated.
See: https://interactivebrokers.github.io/cpwebapi/

Typical gateway URL: ``https://localhost:5000`` (self-signed cert).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from live_cta.core.live import (
    TimestampLike,
    _ensure_datetime_index,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IBKRConfig:
    """Connection and request settings for the IBKR Client Portal API."""

    base_url: str = "https://localhost:5000"
    account_id: str = ""
    verify_ssl: bool = False  # Gateway uses self-signed cert by default
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    # Keepalive ping interval (seconds) to prevent gateway session timeout
    keepalive_interval: int = 55


# ---------------------------------------------------------------------------
# Contract helper
# ---------------------------------------------------------------------------

@dataclass
class IBKRContract:
    """Minimal contract specification for a futures instrument."""

    conid: str
    ticker: str
    exchange: str = "NYMEX"
    sec_type: str = "FUT"
    bar_size: str = "1min"
    # Mapping for expected columns from IBKR bar data
    column_map: Dict[str, str] = field(default_factory=lambda: {
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "TotalVolume",
    })


# ---------------------------------------------------------------------------
# IBKR Tick Data Source
# ---------------------------------------------------------------------------

class IBKRTickDataSource:
    """Live data provider fetching bars from the IBKR Web API.

    Implements the :class:`~CTAFlow.models.evaluation.live.TickDataSource`
    protocol so it can be plugged directly into
    :class:`~CTAFlow.models.evaluation.live.LiveV3FeatureInterface`.

    Parameters
    ----------
    config : IBKRConfig
        Gateway connection settings.
    contracts : dict[str, IBKRContract]
        Mapping of CTAFlow ticker names to IBKR contract specs.
        Example: ``{"NG": IBKRContract(conid="...", ticker="NG")}``
    tz : str
        Target timezone for output data (default ``America/Chicago``).
    """

    def __init__(
        self,
        config: IBKRConfig,
        contracts: Dict[str, IBKRContract],
        tz: str = "America/Chicago",
    ) -> None:
        self.config = config
        self.contracts = contracts
        self.tz = tz
        self.session = self._build_session()
        self._last_keepalive: float = 0.0

    # ------------------------------------------------------------------
    # Session / HTTP helpers
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.verify = self.config.verify_ssl
        retry = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/v1/api{path}"

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Issue GET request and return parsed JSON."""
        url = self._url(path)
        resp = self.session.get(url, params=params, timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: Optional[Dict[str, Any]] = None) -> Any:
        url = self._url(path)
        resp = self.session.post(url, json=json or {}, timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Gateway keepalive
    # ------------------------------------------------------------------

    def tickle(self) -> Dict[str, Any]:
        """Ping the gateway to keep the session alive."""
        result = self._post("/tickle")
        self._last_keepalive = time.monotonic()
        return result

    def ensure_alive(self) -> None:
        """Send a keepalive ping if the interval has elapsed."""
        elapsed = time.monotonic() - self._last_keepalive
        if elapsed >= self.config.keepalive_interval:
            try:
                self.tickle()
            except Exception as exc:
                logger.warning("IBKR keepalive failed: %s", exc)

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def get_accounts(self) -> List[Dict[str, Any]]:
        return self._get("/portfolio/accounts")

    def get_positions(self, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        acct = account_id or self.config.account_id
        return self._get(f"/portfolio/{acct}/positions/0")

    # ------------------------------------------------------------------
    # Contract search
    # ------------------------------------------------------------------

    def search_contract(self, symbol: str, sec_type: str = "FUT") -> List[Dict[str, Any]]:
        """Search for contracts by symbol."""
        return self._post("/iserver/secdef/search", json={
            "symbol": symbol,
            "secType": sec_type,
        })

    def get_contract_details(self, conid: str) -> Dict[str, Any]:
        return self._get(f"/iserver/contract/{conid}/info")

    # ------------------------------------------------------------------
    # Market data snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self, conid: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Request a live market data snapshot.

        Commonly used fields: 31=Last, 84=Bid, 86=Ask, 7295=Volume.
        The IBKR API may require 2 calls for initial snapshot subscription.
        """
        field_str = ",".join(fields or ["31", "84", "86", "7295"])
        result = self._get(
            "/iserver/marketdata/snapshot",
            params={"conids": conid, "fields": field_str},
        )
        # IBKR returns list; first call may be empty (subscription warmup)
        if isinstance(result, list) and result:
            return result[0]
        return result if isinstance(result, dict) else {}

    # ------------------------------------------------------------------
    # Historical bars (core data method)
    # ------------------------------------------------------------------

    def get_history(
        self,
        conid: str,
        period: str = "2d",
        bar_size: str = "1min",
        outside_rth: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical bars from IBKR.

        Parameters
        ----------
        conid : str
            IBKR contract identifier.
        period : str
            Lookback period (e.g. ``"1d"``, ``"2d"``, ``"1w"``).
        bar_size : str
            Bar granularity (e.g. ``"1min"``, ``"5min"``, ``"1h"``).
        outside_rth : bool
            Include data outside regular trading hours.

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, TotalVolume indexed by Datetime.
        """
        data = self._get(
            "/iserver/marketdata/history",
            params={
                "conid": conid,
                "period": period,
                "bar": bar_size,
                "outsideRth": str(outside_rth).lower(),
            },
        )

        bars = data.get("data", [])
        if not bars:
            logger.warning("IBKR returned no bars for conid=%s period=%s", conid, period)
            return pd.DataFrame()

        rows = []
        for bar in bars:
            rows.append({
                "Datetime": pd.to_datetime(bar["t"], unit="ms", utc=True),
                "Open": bar.get("o", np.nan),
                "High": bar.get("h", np.nan),
                "Low": bar.get("l", np.nan),
                "Close": bar["c"],
                "TotalVolume": bar.get("v", 0),
            })

        df = pd.DataFrame(rows)
        return df

    # ------------------------------------------------------------------
    # TickDataSource protocol implementation
    # ------------------------------------------------------------------

    def get_ticks(
        self,
        ticker: str,
        start_time: TimestampLike,
        end_time: TimestampLike,
    ) -> pd.DataFrame:
        """Fetch historical bars and format for CTAFlow live pipeline.

        Satisfies the :class:`TickDataSource` protocol.  Because the IBKR
        Web API does not provide granular bid/ask volume on historical bars,
        ``BidVolume`` and ``AskVolume`` are estimated as half of
        ``TotalVolume``.  For higher-fidelity bid/ask splits, consider
        supplementing with real-time streaming data.

        Parameters
        ----------
        ticker : str
            CTAFlow ticker name (must be registered in ``self.contracts``).
        start_time, end_time : TimestampLike
            Time window to return.

        Returns
        -------
        pd.DataFrame
            Timezone-aware DatetimeIndex with columns:
            ``Close``, ``TotalVolume``, ``BidVolume``, ``AskVolume``,
            ``NumTrades``.
        """
        if ticker not in self.contracts:
            raise KeyError(
                f"Ticker '{ticker}' not found in IBKR contracts. "
                f"Available: {list(self.contracts.keys())}"
            )

        contract = self.contracts[ticker]
        self.ensure_alive()

        # Determine lookback period from the requested window
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        delta_days = max(1, math.ceil((end_ts - start_ts).total_seconds() / 86400))
        period = f"{min(delta_days + 1, 30)}d"

        df = self.get_history(
            conid=contract.conid,
            period=period,
            bar_size=contract.bar_size,
            outside_rth=True,
        )

        if df.empty:
            return pd.DataFrame()

        # IBKR history doesn't split bid/ask volume — estimate 50/50
        df["BidVolume"] = df["TotalVolume"] * 0.5
        df["AskVolume"] = df["TotalVolume"] * 0.5
        df["NumTrades"] = np.where(df["TotalVolume"] > 0, 1, 0)

        df = _ensure_datetime_index(df, self.tz)

        # Filter to requested window
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize(self.tz)
        else:
            start_ts = start_ts.tz_convert(self.tz)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize(self.tz)
        else:
            end_ts = end_ts.tz_convert(self.tz)

        mask = (df.index >= start_ts) & (df.index <= end_ts)
        return df.loc[mask].copy()

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        conid: str,
        side: str,
        quantity: int,
        order_type: str = "MKT",
        price: Optional[float] = None,
        tif: str = "GTC",
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit an order via the IBKR Web API.

        Parameters
        ----------
        conid : str
            Contract ID.
        side : str
            ``"BUY"`` or ``"SELL"``.
        quantity : int
            Number of contracts.
        order_type : str
            ``"MKT"``, ``"LMT"``, ``"STP"``, etc.
        price : float, optional
            Limit/stop price (required for LMT/STP orders).
        tif : str
            Time in force: ``"GTC"``, ``"DAY"``, ``"IOC"``.
        account_id : str, optional
            Override config account ID.

        Returns
        -------
        dict
            IBKR order response.  May contain a confirmation prompt
            (``"id"`` field) that requires a reply call.
        """
        acct = account_id or self.config.account_id
        order_body: Dict[str, Any] = {
            "conid": int(conid),
            "orderType": order_type,
            "side": side.upper(),
            "quantity": quantity,
            "tif": tif,
        }
        if price is not None:
            order_body["price"] = price

        return self._post(
            f"/iserver/account/{acct}/orders",
            json={"orders": [order_body]},
        )

    def confirm_order(self, reply_id: str) -> Dict[str, Any]:
        """Confirm an order that requires user acknowledgement."""
        return self._post(f"/iserver/reply/{reply_id}", json={"confirmed": True})

    def get_live_orders(self) -> Dict[str, Any]:
        """Get all live (open) orders."""
        return self._get("/iserver/account/orders")

    def cancel_order(
        self,
        order_id: str,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel a specific order."""
        acct = account_id or self.config.account_id
        return self._post(f"/iserver/account/{acct}/order/{order_id}", json={
            "action": "cancel",
        })
