# Model Strategy & Architecture

> **Goal:** Predict the full probability distribution of London's daily maximum temperature (Tmax) to price weather markets.

---

## 1. Problem Framing

Tmax market resolves on the **observed Tmax** from an official station (likely Heathrow EGLL or St James's Park) for a given calendar day. The market asks whether Tmax will fall above or below a specific threshold (e.g. "Will Tmax exceed 20°C today?").

This means we do not need a single point forecast — we need a **calibrated predictive distribution**. The model outputs three quantiles `[q10, q50, q90]` which are then converted to a full CDF to price the market and compute betting edge.

---

## 2. Data Sources

### 2.1 Weather Underground (Stream 1 — Observation Stream)

- **What it is:** Live station-level hourly observations from a London WU station.
- **Variables:** Temperature, Dew Point, Humidity, Wind Speed, Pressure.
- **Availability:** Live, updated hourly. Also used for historical training data.
- **Role in model:** Primary real-time signal. Captures what the atmosphere is actually doing this morning at the target location.
- **Limitation:** Point observation — misses large-scale synoptic context. Subject to station errors.

### 2.2 Open-Meteo / UK Met Office 2km NWP Forecast (Stream 2 — NWP Stream)

- **What it is:** Numerical weather prediction output from the UK Met Office 2km UKV model, accessed via Open-Meteo.
- **Variables:** Temperature 2m, Shortwave Radiation (GHI), Direct Radiation, Cloud Cover, Wind Speed, Precipitation Probability, CAPE.
- **Availability:** Daily forecast issued each morning; historical forecast archive accessible via API.
- **Role in model:** Dominant prior for same-day Tmax. NWP 0–24h forecasts for London Tmax have RMSE ~1.5°C. The model uses the full hourly forecast trajectory for the day (06:00–18:00 UTC), not just a summary.
- **Note on solar variables:** Open-Meteo's `shortwave_radiation` (GHI) is equivalent to ERA5 `ssrd` but is already deaccumulated — no manual `diff().clip()` required.

### 2.3 ERA5-Land (Stream 3 — Climatological Prior)

- **What it is:** ECMWF reanalysis, accessed via the DestinE Earth Data Hub Zarr store.
- **Variables downloaded:** `t2m`, `ssrd`, `sshf`, `slhf`, `skt`, `stl1`, `sp`, `u10`, `v10`.
- **Availability:** Historical only (~1950–3 months ago). **Not used live.**
- **Role in model:** Provides a static climatological prior. Precomputed once into a 365-row lookup table (`climate_normals.parquet`), one row per day-of-year (DOY). At inference time, the model looks up the current DOY to retrieve historical norms — no live ERA5 access ever needed.

---

## 3. Climatological Normals Table

Computed by `compute_normals.py` from the ERA5-Land yearly parquets produced by `download_era5.py`.

**Shape:** 365 rows × 16 columns (one row per DOY).

| Column | Description |
|---|---|
| `doy` | Day of year, 1–365 |
| `tmax_mean` | Mean historical Tmax for this DOY |
| `tmax_std` | Standard deviation of Tmax — wide std = uncertain day |
| `tmax_p10`, `tmax_p25` | Lower tail of historical Tmax distribution |
| `tmax_p75`, `tmax_p90` | Upper tail of historical Tmax distribution |
| `tmax_trend` | Linear warming trend (°C/year) from scipy linregress |
| `t2m_mean` | Mean 2m temperature |
| `ssrd_mean` | Mean daily solar energy (J m⁻²) |
| `sshf_mean` | Mean sensible heat flux — proxy for boundary layer mixing |
| `slhf_mean` | Mean latent heat flux — energy partitioning signal |
| `skt_mean` | Mean skin temperature (°C) |
| `stl1_mean` | Mean shallow soil temperature (°C) |
| `sp_mean` | Mean surface pressure (Pa) |
| `u10_mean`, `v10_mean` | Mean 10m wind components (m s⁻¹) |
| `wind_speed_mean` | Mean wind speed magnitude `√(u10² + v10²)` |

A 7-day centred rolling mean is applied to smooth per-DOY noise (configurable via `--smooth_window`).

---

## 4. Model Architecture

### 4.1 Overview

The model is a **three-stream fusion network** that combines:

- A **CNN-BiLSTM** encoding of morning station observations
- A **lightweight CNN-GRU** encoding of the NWP forecast trajectory
- A **learned climatological embedding** looked up by day-of-year

The streams are fused via a **gating mechanism** before passing to the quantile regression head.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STREAM 1 — Observation Stream                                          │
│  WU hourly obs, morning cutoff (B, 6, T_obs)                           │
│       ↓                                                                 │
│  CNN-1D (2 layers, 64ch) → BiLSTM (128 hidden, 2 layers)               │
│       ↓                                                                 │
│  h_obs  (B, 256)                                                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│  STREAM 2 — NWP Forecast Stream                                         │
│  Open-Meteo UKV forecast for today (B, F_nwp, T_forecast)              │
│       ↓                                                                 │
│  CNN-1D (1 layer, 32ch) → GRU (64 hidden, 1 layer)                     │
│       ↓                                                                 │
│  h_nwp  (B, 128)                                                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│  STREAM 3 — Climatological Prior                                        │
│  day_of_year → Embedding(365, d_clim)                                   │
│       ↓                                                                 │
│  2-layer MLP with LayerNorm → h_clim  (B, 64)                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│  FUSION — NWP-Gated Attention                                           │
│                                                                         │
│  Step 1 — Observations correct the NWP prior:                          │
│    gate_obs  = σ( Linear([h_obs, h_nwp]) )          (B, 128)           │
│    h_corr    = gate_obs ⊙ h_nwp                                        │
│             + (1 − gate_obs) ⊙ proj(h_obs)          (B, 128)           │
│                                                                         │
│  Step 2 — Climatology anchors the blend:                               │
│    h_fused   = LayerNorm(concat([h_corr, h_clim]))  (B, 192)           │
│             → Linear(192, 128) → ReLU                                   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────────┐
│  HEAD                                                                   │
│  Linear(128, 64) → ReLU → Dropout(0.2)                                 │
│  Linear(64, 3)  → [q10, q50, q90]                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Design Rationale

**Why NWP leads and observations correct:**
For same-day Tmax prediction, NWP forecast skill is high. The UKV 2km model captures the large-scale forcing that determines the afternoon temperature ceiling. Morning station observations add value by detecting whether the day is tracking warmer or cooler than forecast — particularly through early cloud cover discrepancies or anomalous morning temperatures.

The gate implements this precisely: when observations align with the NWP forecast, the gate balances both streams. When the morning is anomalously cold or warm vs. the forecast, the gate shifts weight toward the observation stream.

**Why a learned embedding for climatology:**
A simple lookup of mean/std would give the model static information but no capacity to learn *how much to trust history* for a given DOY. The `nn.Embedding(365, d_clim)` is initialised from the ERA5 normals matrix (projected to `d_clim` dimensions) and fine-tuned during training. This means the model starts with real physical knowledge and adapts it to the specific forecasting task.

**Why smaller capacity for Stream 2:**
NWP output is smooth, gridded, and already physically consistent. It needs less representational capacity than raw station observations. A single CNN layer and GRU (rather than BiLSTM) is sufficient.

### 4.3 Embedding Initialisation

Rather than random initialisation, the embedding weights are seeded from the ERA5 normals:

```python
normals_matrix = load_and_normalise("climate_normals.parquet")  # (365, F_clim)
with torch.no_grad():
    embedding.weight.data.copy_(
        nn.Linear(F_clim, d_clim)(torch.tensor(normals_matrix, dtype=torch.float32))
    )
```

This accelerates convergence and prevents the embedding from drifting to physically implausible values on DOYs with few training samples (e.g. rare winter dates in a short training set).

---

## 5. Training

### 5.1 Loss Function

**Pinball loss** (quantile regression) for the three quantiles:

```
L = Σ_q  ρ_q(y - ŷ_q)

where ρ_q(u) = u·q        if u ≥ 0
              = u·(q − 1)  if u < 0
```

An optional auxiliary MSE term on q50 vs. the climatological `tmax_mean` is applied early in training (annealed to zero) to regularise the model before it has seen sufficient data:

```
total_loss = pinball_loss + λ · MSE(q50, clim_tmax_mean_for_doy)
```

`λ` starts at 0.1 and decays linearly to 0 over the first 10 epochs.

### 5.2 Optimiser

Adam with **parameter group learning rates**:

| Parameter group | Learning rate |
|---|---|
| `ClimateEmbedding` (pretrained from ERA5) | `1e-4` |
| All other parameters | `3e-4` |

Lower LR for the embedding prevents overwriting the ERA5-derived initialisation before the rest of the network has stabilised.

### 5.3 Training Data Alignment

For each training sample (one day `D`):

| Stream | Data |
|---|---|
| Stream 1 | WU station obs for day `D`, hours 00:00–`cutoff` (e.g. 09:00 UTC) |
| Stream 2 | UKV forecast issued on morning of `D`, valid for day `D` (hourly 06:00–18:00) |
| Stream 3 | ERA5 climate normals for DOY of `D` (static lookup, no download) |
| Target | Observed Tmax on `D` from WU or official station |

**The forecast provenance problem:** Open-Meteo's historical API returns actuals, not what the forecast *said* on the morning of day `D`. Two strategies are used in parallel:

- **Phase 1 (bootstrap):** Use Open-Meteo historical actuals as a forecast proxy. The model will see slightly optimistic NWP inputs during training but the architecture remains correct.
- **Phase 2 (production):** A daily cron job calls the Open-Meteo forecast API each morning and stores the output. This accumulates a ground-truth forecast archive over time, progressively replacing the proxy data in the training set.

---

## 6. Inference Pipeline

At inference time (each morning, before the market closes):

```
1. Fetch WU hourly obs for today up to cutoff hour
2. Fetch Open-Meteo UKV forecast for today (issued this morning)
3. Look up ERA5 climate normals for today's DOY (local file, instant)
4. Run TMaxThreeStream → [q10, q50, q90]
5. Fit skew-normal distribution to the three quantiles
6. Compute P(Tmax > threshold) from the fitted CDF
7. Compare to Polymarket implied probability → compute edge
8. Size bet via fractional Kelly criterion if edge > threshold
```

---

## 7. Betting Layer

### 7.1 CDF Construction

Three quantiles define an under-determined distribution. We fit a **skew-normal** (or Johnson SU) distribution to `[q10, q50, q90]`, which accommodates the slight right-skew of Tmax in warm seasons:

```python
from scipy.stats import skewnorm
# Fit skew-normal parameters to match q10, q50, q90
# → returns loc, scale, skewness
# → evaluate CDF at market threshold
```

### 7.2 Edge and Kelly Sizing

```
model_prob  = P(Tmax > threshold)   from fitted CDF
market_prob = Polymarket implied probability

edge = model_prob − market_prob

# Only bet if edge exceeds minimum threshold (e.g. 5%)
# Kelly fraction = edge / (1 − market_prob)   for binary YES bet
# Bet size = bankroll × f_kelly × fractional_kelly_multiplier
```

A fractional Kelly multiplier (typically 0.25–0.5) is applied to account for model uncertainty and avoid overbetting.

---

## 8. Component Inventory

| Component | Script / Module | Status |
|---|---|---|
| ERA5 download | `download_era5.py` | Complete |
| Climatological normals | `compute_normals.py` | Complete |
| Climate encoder module | `ClimateEncoder` (PyTorch) | Planned |
| Climatological gate module | `ClimatologicalGate` (PyTorch) | Planned |
| NWP stream module | `NWPStream` (PyTorch) | Planned |
| Full model | `TMaxThreeStream` (PyTorch) | Planned |
| Open-Meteo historical fetcher | `fetch_openmeteo_historical.py` | Planned |
| Open-Meteo forecast fetcher (daily cron) | `fetch_openmeteo_forecast.py` | Planned |
| Dataset class (3-stream) | `WeatherDailyDataset` (PyTorch) | Planned |
| CDF post-processor | `fit_distribution.py` | Planned |
| Betting edge calculator | `betting_edge.py` | Planned |

---

## 9. Key Assumptions and Risks

**Stationarity of climatological prior:** The ERA5 normals are computed over a historical window. Climate change causes gradual drift — the `tmax_trend` column captures this, but regime shifts (e.g. sudden urban heat island growth) are not modelled.

**WU station reliability:** Weather Underground data is crowd-sourced from personal weather stations. Data quality varies. Outlier filtering and cross-validation against Open-Meteo actuals is recommended during preprocessing.

**Polymarket resolution station:** The model targets London Tmax generically. If Polymarket resolves on a specific official station (e.g. Heathrow), the WU station should be the nearest available to that reference point.

**Short training set:** UKV historical forecast archive is available from 2017. This gives ~7 years of training data — sufficient but not abundant. The climatological prior embedding is specifically designed to compensate for sparse DOY coverage.