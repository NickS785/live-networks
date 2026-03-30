from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from CTAFlow.models.prep.intraday_continuous import SessionSpec
from live_cta.core.live import InMemoryTickDataSource
from live_cta.core.ng_live import NatGasLiveInterface, ng_default_config
from live_cta.sources.history_backfill_source import HistoricalBackfillDataSource


def _make_ng_bar_history(periods: int = 45) -> pd.DataFrame:
    frames = []
    days = pd.date_range("2026-01-02", periods=periods, freq="D", tz="America/Chicago")
    for day_idx, day in enumerate(days):
        idx = pd.date_range(
            day + pd.Timedelta(hours=2, minutes=30),
            day + pd.Timedelta(hours=15),
            freq="15min",
            tz="America/Chicago",
        )
        steps = np.arange(len(idx), dtype=np.float32)
        close = 3.0 + day_idx * 0.01 + steps * 0.0008 + np.sin(steps / 6.0) * 0.01
        open_ = close - 0.002
        high = close + 0.004
        low = close - 0.004
        volume = np.full(len(idx), 100 + day_idx, dtype=np.float32)
        frames.append(
            pd.DataFrame(
                {
                    "Open": open_,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                },
                index=idx,
            )
        )
    return pd.concat(frames).sort_index()


def _make_interface(source, *, feature_cols=None) -> NatGasLiveInterface:
    cfg = ng_default_config(
        bar_minutes=15,
        target_horizon_minutes=150,
        refresh_interval="15min",
        history_lookback_days=90,
        sessions=[SessionSpec("USA", "02:30", "15:00")],
        tech_lookback=20,
        seq_lookback_bars=8,
        numbars_lookback=4,
        ae_window=10,
    )
    return NatGasLiveInterface(
        cfg,
        source,
        ae_window=10,
        feature_cols=feature_cols,
    )


def test_natgas_interface_supports_bar_only_history() -> None:
    bars = _make_ng_bar_history()
    source = InMemoryTickDataSource({"NG": bars}, tz="America/Chicago")
    interface = _make_interface(source)

    snapshot = interface.get_inference_snapshot(pd.Timestamp("2026-02-10 10:00", tz="America/Chicago"))
    model_inputs = snapshot.to_model_inputs(add_batch_dim=True)

    assert snapshot.raw_ticks.empty
    assert not snapshot.bars.empty
    assert snapshot.sample["tech_features"].shape[0] == 20
    assert snapshot.sample["tech_features"].shape[1] > 0
    assert model_inputs["ae_input"].shape == (1, 10, 12)


def test_natgas_interface_applies_checkpoint_feature_order_and_zero_fill() -> None:
    bars = _make_ng_bar_history()
    source = InMemoryTickDataSource({"NG": bars}, tz="America/Chicago")

    baseline = _make_interface(source)
    baseline_snapshot = baseline.get_inference_snapshot(
        pd.Timestamp("2026-02-10 10:00", tz="America/Chicago")
    )
    base_cols = list(baseline_snapshot.sample["tech_feature_cols"])
    chosen_cols = [base_cols[0], "storage_level", "dd_hdd", "missing_feature"]

    interface = _make_interface(source, feature_cols=chosen_cols)
    interface._maybe_refresh_daily_caches = lambda anchor_ts: None  # type: ignore[method-assign]
    interface._get_daily_context = lambda anchor_ts: {  # type: ignore[method-assign]
        "storage_level": 123.0,
        "dd_hdd": 45.0,
    }

    snapshot = interface.get_inference_snapshot(pd.Timestamp("2026-02-10 10:00", tz="America/Chicago"))
    seq = snapshot.sample["tech_features"]

    assert snapshot.sample["tech_feature_cols"] == chosen_cols
    assert seq.shape == (20, 4)
    assert np.allclose(seq[:, 1], 123.0)
    assert np.allclose(seq[:, 2], 45.0)
    assert np.allclose(seq[:, 3], 0.0)


def test_historical_backfill_source_merges_history_with_live_segment() -> None:
    idx = pd.date_range("2026-02-10 02:30", periods=4, freq="15min", tz="America/Chicago")
    history = pd.DataFrame(
        {
            "Datetime": idx,
            "Open": [1.0, 1.1, 1.2, 1.3],
            "High": [1.1, 1.2, 1.3, 1.4],
            "Low": [0.9, 1.0, 1.1, 1.2],
            "Close": [1.05, 1.15, 1.25, 1.35],
            "Volume": [10, 11, 12, 13],
        }
    )
    with NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        dir=Path.cwd(),
    ) as tmp_file:
        history_path = Path(tmp_file.name)
        history.to_csv(history_path, index=False)

        live_idx = idx[-2:]
        live = pd.DataFrame(
            {
                "Open": [9.0, 9.1],
                "High": [9.2, 9.3],
                "Low": [8.8, 8.9],
                "Close": [9.05, 9.15],
                "Volume": [99, 100],
            },
            index=live_idx,
        )
        live_source = InMemoryTickDataSource({"NG": live}, tz="America/Chicago")
        stitched = HistoricalBackfillDataSource(
            live_source,
            history_map={"NG": history_path},
            tz="America/Chicago",
        )

        result = stitched.get_ticks("NG", idx[0], idx[-1])

        assert len(result) == 4
        assert float(result.loc[idx[-2], "Close"]) == 9.05
        assert float(result.loc[idx[-1], "Close"]) == 9.15
        assert float(result.loc[idx[0], "Close"]) == 1.05
    history_path.unlink(missing_ok=True)
