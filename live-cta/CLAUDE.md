# CLAUDE.md

Concise guide for working on live-cta, a live inference pipeline extracted from CTAFlow. Handles IBKR tick data, cloud storage, feature building, and model inference for CTA futures trading.

## Setup quickstart
- Install in editable mode: `pip install -e .`
- With all optional deps: `pip install -e ".[ibkr,s3,gcs,dashboard,server,dev]"`
- Run tests: `python -m pytest tests/`
- CTAFlow must be installed as a dependency (`pip install -e ../CTAFlow` or equivalent)

## Relationship to CTAFlow
This package is an **extension** of CTAFlow, not a replacement. CTAFlow handles analysis, feature engineering, model architectures, and training. live-cta handles **live inference deployment**: fetching real-time data, building feature tensors, running forward passes, and producing position signals.

### CTAFlow imports that live-cta depends on (do NOT remove from CTAFlow)
- `CTAFlow.data.datasets.tft` (`build_ticker_registry`)
- `CTAFlow.data.datasets.v3_continuous` (`compute_ae_daily_features`, `prepare_vpin_spatial_features`, `rasterize_vpin_to_grid`, `scale_numbars`, `scale_vpin_features`)
- `CTAFlow.data.raw_formatting.intraday_manager` (`read_exported_df`)
- `CTAFlow.features.tick_extractor` (`_compute_profile_levels`)
- `CTAFlow.features.volume.profile` (`MarketProfileExtractor`, `NumberBarsExtractor`)
- `CTAFlow.features.volume.vpin` (`SequenceRasterizer`, `VPINExtractor`)
- `CTAFlow.features.base_extractor` (`ScidBaseExtractor`, lazy import)
- `CTAFlow.models.prep.intraday_continuous` (`ContinuousIntradayPrep`, `SessionSpec`)
- `CTAFlow.models.deep_learning.multi_branch.ng_moe` (`HybridConfig`, `HybridMixtureNetwork`, `MoEConfig`, `NatGasMoE`)
- `CTAFlow.models.deep_learning.training.backtest` (`predictions_to_positions`)

## Architecture snapshot

### Core (`live_cta/core/`)
Extracted from `CTAFlow/models/evaluation/`.

- `live.py`: Protocol `TickDataSource` (`get_ticks(ticker, start_time, end_time) -> DataFrame`). Main class `LiveV3FeatureInterface` builds technical + orderflow features from any tick source. `LiveFeatureSnapshot` holds the computed sample dict and converts to model tensors via `to_model_inputs(add_batch_dim=True)`. `LiveEvaluationConfig` dataclass holds all feature params (VPIN, profile, numbars, raster, AE window). Also provides `ForwardReplayBacktester` for walk-forward backtesting, `TradeLedger`/`TradeRecord` for trade tracking, and `SimulatedTradingServer` for simulated order execution. Tick sources: `InMemoryTickDataSource`, `CsvTickDataSource`, `SierraTickDataSource`.
- `ng_live.py`: `NatGasLiveInterface` extends `LiveV3FeatureInterface` with daily context injection (EIA storage, population-weighted weather, VAE regime features). `DailyContextPaths` dataclass points to cached context files. `ng_default_config(**overrides)` returns pre-tuned `LiveEvaluationConfig` for NG 30-min inference. Self-contained weather pipeline: `process_weather_features()` computes 13 `dd_*` features (HDD/CDD, 7d rolling, spline basis, weighted avg temp) without requiring macrOS-Int at inference time. `WEATHER_FEATURE_COLS` lists the 13 feature names. `REGIME_FEATURES` lists the 12 features expected by the VAE regime encoder.

### Sources (`live_cta/sources/`)
Extracted from `CTAFlow/data/ext/`.

- `ibkr_client.py`: `IBKRTickDataSource` fetches bars from the IBKR Client Portal Web API. Implements `TickDataSource` protocol. `IBKRConfig` (gateway URL, SSL, retries, keepalive). `IBKRContract` (conid, ticker, exchange, bar_size). Also provides `place_order()`, `confirm_order()`, `get_positions()`, `cancel_order()` for order management. Session keepalive via `tickle()`/`ensure_alive()`. IBKR history doesn't split bid/ask volume; `get_ticks()` estimates 50/50 split.
- `s3_tick_source.py`: `S3TickDataSource` fetches bars from S3-compatible storage (AWS, RunPod, MinIO). Uses `AWSClient` internally. `S3TickerSpec` maps ticker to S3 key + format.
- `gcs_tick_source.py`: `GCSTickDataSource` fetches bars from Google Cloud Storage. `GCSConfig` (bucket, prefix, project, credentials). `GCSTickerSpec` maps ticker to GCS object path + format. Lazy client init.

### Storage (`live_cta/storage/`)
Extracted from `CTAFlow/data/storage/`. No CTAFlow imports.

- `aws_client.py`: `AWSClient` wraps boto3 for S3 operations with caching, configurable endpoints. `S3Config` dataclass (bucket, region, endpoint_url, credentials, prefix). Methods: `upload_file()`, `download_file()`, `object_exists()`, `_add_prefix()`. Attribute for direct access: `s3_client` (the boto3 client).
- `model_manager.py`: `ModelManager` handles model versioning, checkpointing, metadata. `ModelMode` enum: TRAINING, BACKTEST, PRODUCTION. Methods: `save_model()`, `load_model()`, `save_checkpoint()`, `load_checkpoint()`.
- `storage_backend.py`: ABC `StorageBackend` with `LocalStorage` and `S3Storage` implementations. Pluggable interface: `exists()`, `upload_file()`, `download_file()`, `list_files()`, `delete_file()`, `get_metadata()`.

### Dashboard (`live_cta/dashboard/`)
Extracted from `CTAFlow/data/utils/`. No CTAFlow imports.

- `s3_client.py`: `S3Client` for dashboard-specific S3 operations (`list_files`, `download_model`, `load_parquet`).
- `parquet_handler.py`: `ParquetHandler` saves/loads predictions and backtests as parquet.
- `data_loader.py`: `DataLoader` unifies S3Client + ParquetHandler for dashboard data access.

## Scripts

All scripts import model architectures from CTAFlow (`HybridMixtureNetwork`, `NatGasMoE`) and live infrastructure from `live_cta`.

- `scripts/run_ng_inference.py`: Main inference runner. IBKR live ticks + daily context (local/S3/GCS/API) -> `NatGasLiveInterface` -> `HybridMixtureNetwork` -> position signal. Supports `--refresh-context` to fetch fresh EIA + weather from macrOS-Int APIs. `--loop` for continuous operation with daily auto-refresh. Output: JSON with `pred_return`, `pred_std`, `position`, `discrete_position`.
- `scripts/inference_server.py`: FastAPI HTTP server wrapping the inference cycle. Runs on RunPod (S3 backend) or Cloud Run (GCS backend). Endpoints: `POST /infer`, `GET /health`. Also supports `--once` for single-shot mode.
- `scripts/local_feeder.py`: IBKR -> parquet -> S3/GCS upload loop. Runs on the local machine. Fetches bars, saves parquet, uploads on fixed interval (default 300s).
- `scripts/ng_live_runner.py`: Production live-trading runner with order execution. 30-min cadence, trading hours check (08:30-16:00 CT), graceful shutdown via SIGINT/SIGTERM. `--dry-run` logs orders without submitting. `--max-contracts` caps position size.

### CLI entry points (after `pip install -e .`)
- `ng-inference` -> `run_ng_inference:main`
- `ng-feeder` -> `local_feeder:main`
- `ng-server` -> `inference_server:main`

## Key environment variables
- `IBKR_BASE_URL` (default `https://localhost:5000`)
- `IBKR_ACCOUNT_ID`
- `NG_CONID` (IBKR contract ID for NG front month)
- `S3_BUCKET`, `S3_ENDPOINT`, `S3_PREFIX` (S3/RunPod config)
- `GCS_BUCKET`, `GCS_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS` (GCS config)
- `EIA_API_KEY`, `NCEI_TOKEN` (for `--refresh-context` API fetch)
- `MACROSINT_PATH` (path to macrOS-Int, default `~/PycharmProjects/macrOS-Int`)
- `CTAFLOW_CACHE_DIR` (local cache, default `~/.ctaflow/live_context`)

## Data flow
```
Local machine                      Cloud VM (RunPod / Cloud Run)
--------------                     ----------------------------
IBKR Gateway                       S3/GCS bucket
    |                                  |
    v                                  v
local_feeder.py                    inference_server.py
  IBKR -> parquet -> upload          download ticks
                                       |
                                       v
    OR direct:                     NatGasLiveInterface
    run_ng_inference.py              build features
      IBKR live ticks                  |
      + cloud/local daily context      v
              |                    HybridMixtureNetwork
              v                      forward pass
         NatGasLiveInterface           |
           build features              v
              |                    position signal
              v                    (JSON output or order)
         HybridMixtureNetwork
           forward pass
              |
              v
         position signal
         (JSON / IBKR order)
```

## Testing
- Unit tests mock IBKR HTTP responses via `monkeypatch`
- `test_ibkr_client.py`: Verifies history parsing, timezone handling (epoch ms -> UTC -> CT), bid/ask volume estimation
- `test_live_evaluation_interface.py`: Verifies 30-min refresh cadence, V3-style feature snapshot structure, trade ledger lifecycle, forward replay backtester
- Live smoke test (`test_ibkr_live_gateway_smoke`) requires env vars: `CTAFLOW_IBKR_GATEWAY_URL`, `CTAFLOW_IBKR_CONID`

## Working notes
- `TickDataSource` is a protocol, not an ABC. Any object with `get_ticks(ticker, start_time, end_time)` works.
- IBKR Web API gateway uses a self-signed cert; `verify_ssl=False` is the default.
- IBKR history bars arrive as epoch milliseconds in UTC. `get_ticks()` converts to the target timezone.
- IBKR doesn't provide bid/ask volume split on historical bars. `BidVolume` and `AskVolume` are estimated as 50% of `TotalVolume`. For higher fidelity, supplement with real-time streaming.
- Weather features are self-contained in `ng_live.py` (inlined from macrOS-Int). macrOS-Int is only needed for `--refresh-context` fresh API fetch, not for inference.
- `SplineTransformer` (scikit-learn) is optional. If not available, spline HDD basis features are skipped.
- The `AWSClient` attribute for direct boto3 access is `s3_client` (not `s3`). The method to add prefix is `_add_prefix()` (not `_full_key()`).
- `LiveFeatureSnapshot.sample` dict keys: `tech_features`, `tech_len`, `seq_vpin`, `seq_vpin_len`, `ae_input`, `fused_spatial` (if enabled), `numbars_recent`, `vpin_raster_recent`, `ticker_id`, `asset_class_id`, `asset_subclass_id`, `target_end_ts`.
- `HybridMixtureNetwork` forward signature: `(x_seq: (B,L,F), ae_input: (B,ae_window,f_ae))` returns dict with `pred_return`, `pred_std`, `class_logits`, `position`, optionally `mdn_pred_return`, `mdn_pred_std`, `z_regime`, `router_weights`.
