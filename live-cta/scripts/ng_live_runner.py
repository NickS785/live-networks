#!/usr/bin/env python
"""Production live-trading runner for NG HybridMixtureNetwork / NatGasMoE.

Connects to the IBKR Client Portal API Gateway, fetches live bars,
builds features via :class:`NatGasLiveInterface`, runs inference on a
30-minute cadence, and optionally submits orders.

Usage
-----
::

    python -m CTAFlow.scripts.ng_live_runner \\
        --checkpoint ng_hybrid_intraday_best.pth \\
        --conid 583120379 \\
        --account U1234567 \\
        --gateway-url https://localhost:5000 \\
        --dry-run

Environment
-----------
Requires the IBKR Client Portal Gateway to be running and authenticated.
Daily EIA/weather caches should be refreshed by a separate cron job
and placed at the paths specified via ``--eia-path`` / ``--weather-path``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from live_cta.sources.ibkr_client import IBKRConfig, IBKRContract, IBKRTickDataSource
from live_cta.core.ng_live import (
    DailyContextPaths,
    NatGasLiveInterface,
    ng_default_config,
)

logger = logging.getLogger("ng_live_runner")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    model_type: str = "hybrid",
) -> torch.nn.Module:
    """Load a trained model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``.pth`` checkpoint file.
    model_type : str
        ``"hybrid"`` for :class:`HybridMixtureNetwork` or
        ``"moe"`` for :class:`NatGasMoE`.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if model_type == "hybrid":
        from CTAFlow.models.deep_learning.multi_branch.ng_moe import (
            HybridConfig,
            HybridMixtureNetwork,
        )
        config = HybridConfig(**ckpt["config"])
        model = HybridMixtureNetwork(config)
    elif model_type == "moe":
        from CTAFlow.models.deep_learning.multi_branch.ng_moe import (
            MoEConfig,
            NatGasMoE,
        )
        config = MoEConfig(**ckpt["config"])
        model = NatGasMoE(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded %s model from %s", model_type, checkpoint_path)
    return model


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------

class GracefulExit:
    """Handles SIGINT/SIGTERM for clean shutdown."""

    def __init__(self) -> None:
        self.should_exit = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum: int, frame: object) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        self.should_exit = True


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def execute_signal(
    ibkr: IBKRTickDataSource,
    conid: str,
    account_id: str,
    target_exposure: float,
    current_position: int,
    max_contracts: int,
    dry_run: bool = True,
) -> None:
    """Convert model exposure signal to IBKR order.

    Parameters
    ----------
    target_exposure : float
        Model output in [-1, 1] range.
    current_position : int
        Current net position in contracts.
    max_contracts : int
        Maximum position size.
    dry_run : bool
        If True, log the order but don't submit.
    """
    target_contracts = int(round(target_exposure * max_contracts))
    delta = target_contracts - current_position

    if delta == 0:
        logger.info("No position change needed (target=%d, current=%d)", target_contracts, current_position)
        return

    side = "BUY" if delta > 0 else "SELL"
    qty = abs(delta)

    logger.info(
        "Order signal: %s %d contracts (target=%d, current=%d, exposure=%.3f)",
        side,
        qty,
        target_contracts,
        current_position,
        target_exposure,
    )

    if dry_run:
        logger.info("[DRY RUN] Order not submitted")
        return

    try:
        result = ibkr.place_order(
            conid=conid,
            side=side,
            quantity=qty,
            order_type="MKT",
            account_id=account_id,
        )
        logger.info("Order submitted: %s", result)

        # Handle confirmation prompts
        if isinstance(result, list):
            for item in result:
                if "id" in item:
                    confirm = ibkr.confirm_order(str(item["id"]))
                    logger.info("Order confirmed: %s", confirm)

    except Exception as exc:
        logger.error("Order submission failed: %s", exc)


def get_current_position(ibkr: IBKRTickDataSource, conid: str) -> int:
    """Query current position for the given contract."""
    try:
        positions = ibkr.get_positions()
        for pos in positions:
            if str(pos.get("conid")) == str(conid):
                return int(pos.get("position", 0))
    except Exception as exc:
        logger.warning("Failed to fetch positions: %s", exc)
    return 0


# ---------------------------------------------------------------------------
# Inference cycle
# ---------------------------------------------------------------------------

def run_inference_cycle(
    model: torch.nn.Module,
    interface: NatGasLiveInterface,
    ibkr: IBKRTickDataSource,
    conid: str,
    account_id: str,
    max_contracts: int,
    dry_run: bool,
) -> None:
    """Execute one full inference cycle: fetch data -> predict -> act."""
    now = pd.Timestamp.now(tz="America/Chicago")
    logger.info("=== Inference cycle at %s ===", now.strftime("%Y-%m-%d %H:%M:%S %Z"))

    try:
        # 1. Build feature snapshot
        snapshot = interface.get_inference_snapshot(now)
        inputs = snapshot.to_model_inputs(add_batch_dim=True)

        # 2. Run model inference
        with torch.no_grad():
            out = model(inputs["tech_features"], inputs["ae_input"])

        pred_return = out["pred_return"].item()
        pred_std = out.get("pred_std", torch.tensor(0.0)).item()

        logger.info(
            "Prediction: return=%.5f, std=%.5f, anchor=%.4f",
            pred_return,
            pred_std,
            snapshot.anchor_close,
        )

        # 3. Log regime info if available
        if "z_regime" in out:
            z = out["z_regime"].squeeze()
            logger.info("Regime latent (first 4): %s", z[:4].tolist())

        if "router_weights" in out:
            logger.info("Router weights: %s", out["router_weights"].squeeze().tolist())

        # 4. Execute position signal
        if "position" in out:
            target_exposure = out["position"].item()
            logger.info("Target exposure: %.3f", target_exposure)
        else:
            # Map continuous return prediction to exposure
            # Simple threshold: long if pred > 0, short if < 0, scaled by confidence
            confidence = min(abs(pred_return) / (pred_std + 1e-6), 1.0)
            target_exposure = confidence if pred_return > 0 else -confidence
            logger.info("Derived exposure: %.3f (confidence=%.3f)", target_exposure, confidence)

        current_pos = get_current_position(ibkr, conid)

        execute_signal(
            ibkr=ibkr,
            conid=conid,
            account_id=account_id,
            target_exposure=target_exposure,
            current_position=current_pos,
            max_contracts=max_contracts,
            dry_run=dry_run,
        )

        # 5. Record prediction in trade ledger
        interface.record_prediction(
            prediction=pred_return,
            task="regression",
            position=target_exposure,
            meta={
                "pred_std": pred_std,
                "anchor_close": snapshot.anchor_close,
                "model_output": {
                    k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else str(v)
                    for k, v in out.items()
                    if k not in ("z_regime", "router_weights", "ae_losses", "mdn_params")
                },
            },
        )

    except Exception as exc:
        logger.error("Inference cycle failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def is_trading_hours(now: pd.Timestamp) -> bool:
    """Check if current time falls within NG RTH (08:30-16:00 CT)."""
    t = now.time()
    return pd.Timestamp("08:30").time() <= t <= pd.Timestamp("16:00").time()


def next_half_hour_boundary(now: pd.Timestamp) -> pd.Timestamp:
    """Return the next :00 or :30 boundary."""
    floored = now.floor("30min")
    nxt = floored + pd.Timedelta(minutes=30)
    return nxt


def sleep_until(target: pd.Timestamp) -> None:
    """Sleep until the target timestamp, checking every second."""
    while True:
        now = pd.Timestamp.now(tz=target.tz)
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting NG Live Runner")
    logger.info("  Checkpoint: %s", args.checkpoint)
    logger.info("  Model type: %s", args.model_type)
    logger.info("  IBKR Gateway: %s", args.gateway_url)
    logger.info("  ConID: %s", args.conid)
    logger.info("  Account: %s", args.account)
    logger.info("  Max contracts: %d", args.max_contracts)
    logger.info("  Dry run: %s", args.dry_run)

    # 1. Load model
    model = load_model(args.checkpoint, args.model_type)

    # 2. Setup IBKR data source
    ibkr_config = IBKRConfig(
        base_url=args.gateway_url,
        account_id=args.account,
        verify_ssl=not args.no_verify_ssl,
    )
    contracts = {
        "NG": IBKRContract(conid=args.conid, ticker="NG"),
    }
    ibkr_source = IBKRTickDataSource(
        config=ibkr_config,
        contracts=contracts,
        tz="America/Chicago",
    )

    # 3. Setup live interface
    daily_paths = DailyContextPaths(
        eia_storage_path=args.eia_path,
        weather_path=args.weather_path,
        daily_features_path=args.daily_features_path,
    )

    live_config = ng_default_config(
        bar_minutes=args.bar_minutes,
        target_horizon_minutes=args.target_horizon,
        refresh_interval=f"{args.bar_minutes}min",
    )

    interface = NatGasLiveInterface(
        config=live_config,
        data_source=ibkr_source,
        daily_paths=daily_paths,
        ae_window=args.ae_window,
    )

    # 4. Run loop
    exit_handler = GracefulExit()

    logger.info("Entering main loop. Press Ctrl+C to stop.")
    while not exit_handler.should_exit:
        now = pd.Timestamp.now(tz="America/Chicago")

        if not is_trading_hours(now):
            logger.debug("Outside trading hours (%s), sleeping...", now.strftime("%H:%M"))
            time.sleep(60)
            continue

        # Sleep until next 30-min boundary
        target = next_half_hour_boundary(now)
        logger.info("Next inference at %s", target.strftime("%H:%M:%S"))
        sleep_until(target)

        if exit_handler.should_exit:
            break

        # Keep gateway session alive
        ibkr_source.ensure_alive()

        run_inference_cycle(
            model=model,
            interface=interface,
            ibkr=ibkr_source,
            conid=args.conid,
            account_id=args.account,
            max_contracts=args.max_contracts,
            dry_run=args.dry_run,
        )

    logger.info("NG Live Runner stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NG Live Trading Runner with IBKR Web API",
    )
    # Model
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument(
        "--model-type",
        choices=["hybrid", "moe"],
        default="hybrid",
        help="Model architecture (default: hybrid)",
    )

    # IBKR connection
    parser.add_argument(
        "--gateway-url",
        default="https://localhost:5000",
        help="IBKR Client Portal Gateway URL",
    )
    parser.add_argument("--conid", required=True, help="IBKR contract ID for NG futures")
    parser.add_argument("--account", required=True, help="IBKR account ID")
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL verification (for self-signed gateway cert)",
    )

    # Trading
    parser.add_argument("--max-contracts", type=int, default=1, help="Max position size")
    parser.add_argument("--dry-run", action="store_true", help="Log signals without placing orders")

    # Daily data paths
    parser.add_argument("--eia-path", default=None, help="Path to EIA storage cache file")
    parser.add_argument("--weather-path", default=None, help="Path to weather (HDD/CDD) cache file")
    parser.add_argument(
        "--daily-features-path",
        default=None,
        help="Path to pre-computed daily features cache",
    )

    # Inference settings
    parser.add_argument("--bar-minutes", type=int, default=30, help="Bar size in minutes")
    parser.add_argument("--target-horizon", type=int, default=60, help="Prediction horizon in minutes")
    parser.add_argument("--ae-window", type=int, default=21, help="VAE regime lookback window")

    # Logging
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args()


if __name__ == "__main__":
    main()
