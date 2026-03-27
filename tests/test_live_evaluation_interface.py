from __future__ import annotations

import numpy as np
import pandas as pd

from live_cta.core import (
    ForwardReplayBacktester,
    InMemoryTickDataSource,
    LiveEvaluationConfig,
    LiveV3FeatureInterface,
    SimulatedOrder,
)
from CTAFlow.models.prep.intraday_continuous import SessionSpec


def _make_ticks() -> pd.DataFrame:
    frames = []
    base_days = pd.date_range("2026-02-02", periods=4, freq="D", tz="America/Chicago")
    for day_idx, day in enumerate(base_days):
        idx = pd.date_range(
            day + pd.Timedelta(hours=8),
            day + pd.Timedelta(hours=16),
            freq="1min",
            tz="America/Chicago",
        )
        minutes = np.arange(len(idx), dtype=np.float32)
        close = 70.0 + day_idx * 0.35 + minutes * 0.0025 + np.sin(minutes / 15.0) * 0.03
        ask = np.full(len(idx), 40.0 + day_idx, dtype=np.float32)
        bid = np.full(len(idx), 30.0 + day_idx, dtype=np.float32)
        total = ask + bid
        frames.append(
            pd.DataFrame(
                {
                    "Close": close,
                    "AskVolume": ask,
                    "BidVolume": bid,
                    "TotalVolume": total,
                    "NumTrades": np.full(len(idx), 3, dtype=np.int32),
                },
                index=idx,
            )
        )
    return pd.concat(frames).sort_index()


def _make_interface(use_fused_spatial: bool = True) -> LiveV3FeatureInterface:
    ticks = _make_ticks()
    source = InMemoryTickDataSource({"CL": ticks}, tz="America/Chicago")
    cfg = LiveEvaluationConfig(
        ticker="CL",
        tick_size=0.01,
        tz="America/Chicago",
        refresh_interval="30min",
        history_lookback_days=14,
        sessions=[SessionSpec("USA", "08:30", "16:00")],
        bar_minutes=5,
        target_horizon_minutes=30,
        tech_lookback=24,
        seq_lookback_bars=16,
        numbars_lookback=8,
        use_fused_spatial=use_fused_spatial,
        vpin_bucket_volume=300,
        vpin_window=12,
        vpin_start_time="08:30",
        vpin_end_time="16:00",
        profile_start_time="08:00",
        profile_end_time="09:30",
        num_bars_start_time="08:30",
        num_bars_end_time="16:00",
        num_bars_interval="15min",
        num_bars_levels=8,
        raster_num_bars=6,
        raster_interval_mins=30,
        raster_bins=32,
        ae_window=4,
    )
    return LiveV3FeatureInterface(cfg, source)


def test_live_refresh_uses_30_minute_buckets() -> None:
    interface = _make_interface()

    snap_1 = interface.refresh(pd.Timestamp("2026-02-05 10:31", tz="America/Chicago"))
    assert interface.last_refresh_time == pd.Timestamp("2026-02-05 10:30", tz="America/Chicago")
    assert not interface.should_refresh(pd.Timestamp("2026-02-05 10:44", tz="America/Chicago"))

    snap_2 = interface.maybe_refresh(pd.Timestamp("2026-02-05 10:44", tz="America/Chicago"))
    assert snap_2 is snap_1

    assert interface.should_refresh(pd.Timestamp("2026-02-05 11:00", tz="America/Chicago"))
    snap_3 = interface.maybe_refresh(pd.Timestamp("2026-02-05 11:00", tz="America/Chicago"))
    assert snap_3 is not snap_1
    assert interface.last_refresh_time == pd.Timestamp("2026-02-05 11:00", tz="America/Chicago")


def test_live_snapshot_recreates_v3_style_payload() -> None:
    interface = _make_interface(use_fused_spatial=True)
    snapshot = interface.refresh(pd.Timestamp("2026-02-05 11:00", tz="America/Chicago"))

    sample = snapshot.sample
    assert sample["tech_features"].shape == (24, sample["tech_features"].shape[1])
    assert sample["seq_vpin"].ndim == 2
    assert sample["seq_vpin_len"] > 0
    assert sample["ae_input"].shape == (4, 4)
    assert sample["fused_spatial"].shape[0] > 0
    assert sample["fused_spatial"].shape[1] == 7
    assert sample["fused_spatial"].shape[2] == 17
    assert snapshot.anchor_ts < snapshot.target_end_ts

    tensors = snapshot.to_model_inputs(add_batch_dim=False)
    assert tensors["tech_features"].shape[0] == 24
    assert tensors["fused_spatial"].shape[1:] == sample["fused_spatial"].shape[1:]


def test_trade_ledger_closes_after_target_horizon() -> None:
    interface = _make_interface(use_fused_spatial=False)
    interface.refresh(pd.Timestamp("2026-02-05 10:30", tz="America/Chicago"))
    record = interface.record_prediction(0.25, task="regression", threshold=0.0)
    assert record.status == "open"

    interface.refresh(pd.Timestamp("2026-02-05 11:30", tz="America/Chicago"))
    frame = interface.trade_ledger.to_frame()
    assert len(frame) == 1
    assert frame.iloc[0]["status"] == "closed"
    assert np.isfinite(frame.iloc[0]["realized_return"])
    assert interface.trade_ledger.summary()["n_trades"] == 1.0


def test_forward_replay_backtest_stays_backward_looking_and_executes_orders() -> None:
    interface = _make_interface(use_fused_spatial=True)
    runner = ForwardReplayBacktester(interface)

    def decision_fn(snapshot, _server):
        ts = snapshot.anchor_ts
        if ts.hour == 9 and ts.minute == 0:
            return SimulatedOrder(target_position=1.0, prediction=0.9)
        if ts.hour == 10 and ts.minute == 0:
            return {"target_position": 0.0, "prediction": 0.0}
        return None

    result = runner.run(
        start_time=pd.Timestamp("2026-02-05 09:00", tz="America/Chicago"),
        end_time=pd.Timestamp("2026-02-05 10:30", tz="America/Chicago"),
        decision_fn=decision_fn,
        flatten_at_end=True,
    )

    assert not result.steps.empty
    result.assert_backward_looking()
    assert result.steps["backward_looking"].all()
    assert len(result.fills) >= 2
    assert len(result.trades) >= 1
    assert np.isfinite(result.equity_curve["equity"].iloc[-1])
