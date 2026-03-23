"""
Combined CDF
    All draws are weighted (ensembles more heavily than deterministic)
    and fed into a KDE to produce a smooth, full CDF.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde

def plot_forecast_distribution(mu: float, std: float, strikes: list = None):
    """
    Plots PDF and CDF together to visualize the betting landscape.
    """
    # Create range for X axis (4 standard deviations)
    x = np.linspace(mu - 4*std, mu + 4*std, 500)
    pdf = norm.pdf(x, mu, std)
    cdf = norm.cdf(x, mu, std)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # PDF Plot (Likelihood)
    ax1.plot(x, pdf, color='tab:blue', lw=2, label='PDF (Likelihood)')
    ax1.fill_between(x, pdf, color='tab:blue', alpha=0.2)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Density', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # CDF Plot (Cumulative Prob)
    ax2 = ax1.twinx()
    ax2.plot(x, cdf, color='tab:red', lw=2, ls='--', label='CDF (Cumulative)')
    ax2.set_ylabel('Cumulative Probability', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add vertical lines for market strikes if provided
    if strikes:
        for s in strikes:
            ax1.axvline(x=s, color='gray', alpha=0.5, linestyle=':')
            prob = norm.cdf(s, mu, std)
            ax2.scatter(s, prob, color='tab:red')
            ax2.annotate(f'{prob:.1%}', (s, prob), textcoords="offset points", xytext=(0,10), ha='center', color='tab:red')

    plt.title(f'Weather Market Fusion Model\n mean={mu:.2f}, std={std:.2f}')
    fig.tight_layout()
    plt.show()


def calc_mean_std_deterministic_forecast(temperature:np.ndarray,
                                        weights:np.ndarray):
    """
    compute mean and weighted std
    """
    # weighted mean
    mu_w = np.average(temperature,
                        weights=weights)
    
    # weight variance and std
    variance_w = np.average((temperature - mu_w)**2,
                            weights=weights
                            )
    
    # std
    std_w = np.sqrt(variance_w)

    return mu_w, std_w


def calculate_market_edge(mu: float,
                         std: float,
                            bin_low: float,
                            bin_high: float,
                            market_price: float):
    """
    Calculates the probability of a specific temperature bin and the edge vs market.
    """
    # Model probability for the bin [low, high]
    model_prob = norm.cdf(bin_high, mu, std) - norm.cdf(bin_low, mu, std)
    edge = model_prob - market_price
    
    return model_prob, edge


def calculate_kelly_size(model_prob: float,
                            market_price: float,
                            kelly_fraction: float = 0.25):
    """
    Computes the Kelly Criterion fraction.
    kelly_fraction: Use 'Fractional Kelly' (e.g., 0.25) to avoid over-betting.
    """
    if model_prob <= market_price:
        return 0.0  # No edge, no bet
    
    # b = decimal odds - 1. For Polymarket, odds are 1/price
    b = (1 / market_price) - 1
    p = model_prob
    q = 1 - p
    
    f_star = (b * p - q) / b
    
    # Apply fractional Kelly (e.g., Quarter-Kelly) for safety
    return max(0, f_star * kelly_fraction)



def calculate_ensemble_distribution(member_values: np.ndarray):
    """
    Fits a KDE to the ensemble members.
    Returns the KDE object which acts as the PDF/CDF generator.

    Args:
        member_values: np.ndarray.
    """
    # Remove NaNs if any
    data = member_values[~np.isnan(member_values)]
    
    # Fit Kernel Density Estimator
    kde = gaussian_kde(data)
    
    # To get CDF from KDE, we integrate the PDF. 
    # Scipy KDE doesn't have a direct .cdf(), so we use .integrate_box_1d
    return kde


def get_kde_prob_in_bin(kde, low, high):
    """
    Returns the probability P(low < T < high) using the KDE.
    """
    return kde.integrate_box_1d(low, high)


def plot_ensemble_vs_market(kde, member_values, strikes):
    """
    Plots the actual ensemble histogram, the KDE, and the CDF.
    """
    x = np.linspace(min(member_values)-2, max(member_values)+2, 500)
    pdf_values = kde.evaluate(x)
    
    # Calculate CDF by integrating from -infinity to each x
    cdf_values = [kde.integrate_box_1d(-np.inf, val) for val in x]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Histogram of raw ensemble members
    ax1.hist(member_values, bins=30, density=True, alpha=0.3, color='gray', label='Raw Members')
    # Plot KDE PDF
    ax1.plot(x, pdf_values, color='tab:blue', lw=3, label='KDE (Model Likelihood)')
    ax1.set_ylabel('Density', color='tab:blue')

    # Plot CDF
    ax2 = ax1.twinx()
    ax2.plot(x, cdf_values, color='tab:red', lw=2, ls='--', label='CDF')
    ax2.set_ylabel('Cumulative Probability', color='tab:red')

    if strikes:
        for s in strikes:
            ax1.axvline(x=s, color='black', alpha=0.3, ls=':')
            # Label the CDF value at each strike
            p_below = kde.integrate_box_1d(-np.inf, s)
            ax2.scatter(s, p_below, color='tab:red')
            ax2.annotate(f'{p_below:.1%}', (s, p_below), xytext=(5,5), textcoords='offset points', color='red')

    plt.title("Ensemble KDE Distribution (N=217)")
    plt.show()
-----------------
ENSEMBLE


# ============================================================================
# CDF construction
# ============================================================================

def build_weighted_cdf(
    det_vals: dict[str, float],         # {model_name: T_max}
    det_weights: dict[str, float],      # {model_name: weight}
    ens_draws: dict[str, np.ndarray],   # {model_name: array of member T_max}
    ens_weights: dict[str, float],      # {model_name: weight_per_member}
    x_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a weighted KDE CDF from deterministic + ensemble draws.

    Returns
    -------
    all_vals    : raw array of all draws (unweighted)
    all_weights : corresponding weights
    kde_cdf     : smoothed CDF evaluated on x_grid
    emp_cdf     : empirical step CDF (sorted vals, probs)
    """
    vals, weights = [], []

    for name, v in det_vals.items():
        if not np.isnan(v):
            vals.append(v)
            weights.append(det_weights.get(name, 1.0))

    for name, arr in ens_draws.items():
        w = ens_weights.get(name, 0.7)
        for v in arr:
            if not np.isnan(v):
                vals.append(v)
                weights.append(w)

    if len(vals) < 2:
        return None, None, None, None

    vals = np.array(vals)
    weights = np.array(weights)
    weights = weights / weights.sum()   # normalise

    # --- Weighted KDE ---
    # scipy gaussian_kde doesn't natively support weights, so we use
    # a simple trick: repeat values proportional to their weight,
    # scaled to ~500 pseudo-samples to keep bandwidth sensible.
    n_pseudo = 500
    counts = np.round(weights * n_pseudo).astype(int)
    counts = np.maximum(counts, 1)
    pseudo_samples = np.repeat(vals, counts)

    kde = gaussian_kde(pseudo_samples, bw_method="scott")
    kde_pdf = kde(x_grid)
    kde_cdf = np.cumsum(kde_pdf) * (x_grid[1] - x_grid[0])
    kde_cdf = kde_cdf / kde_cdf[-1]

    # --- Empirical CDF (unweighted for reference) ---
    sorted_idx = np.argsort(vals)
    emp_vals = vals[sorted_idx]
    emp_w = weights[sorted_idx]
    emp_cdf = np.cumsum(emp_w)
    emp_cdf = emp_cdf / emp_cdf[-1]

    return vals, weights, kde_cdf, (emp_vals, emp_cdf)


# ============================================================================
# Betting edge
# ============================================================================

def compute_edge(
    x_grid: np.ndarray,
    kde_cdf: np.ndarray,
    threshold: float,
    market_prob: float,
) -> dict:
    """P(T_max > threshold) from KDE CDF vs market implied probability."""
    model_prob = float(1.0 - np.interp(threshold, x_grid, kde_cdf))
    edge = model_prob - market_prob
    kelly = edge / (1.0 - market_prob) if 0 < market_prob < 1 else np.nan
    return {
        "model_prob": model_prob,
        "market_prob": market_prob,
        "edge": edge,
        "kelly_fraction": kelly,
    }



-------------------------
# ------------------------------------------------------------------
    # 5. Build CDF per day and plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(6 * len(target_dates), 10))
    gs = gridspec.GridSpec(2, len(target_dates), height_ratios=[2, 1], hspace=0.4)

    cdf_store: dict[date, tuple] = {}

    for col_idx, d in enumerate(target_dates):
        ax_cdf  = fig.add_subplot(gs[0, col_idx])
        ax_box  = fig.add_subplot(gs[1, col_idx])

        # Gather inputs for this date
        d_det_vals = {
            name: det_maxima[name][d]
            for name in det_maxima
            if not np.isnan(det_maxima[name].get(d, np.nan))
        }
        d_det_weights = {name: DET_MODELS[name]["weight"] for name in d_det_vals}

        d_ens_draws = {
            name: ens_maxima[name][d]
            for name in ens_maxima
            if not np.all(np.isnan(ens_maxima[name].get(d, [np.nan])))
        }
        d_ens_weights = {
            name: ENS_MODELS[name]["weight_per_member"] for name in d_ens_draws
        }

        # Collect all raw values for axis range
        all_raw = list(d_det_vals.values())
        for arr in d_ens_draws.values():
            all_raw.extend(arr[~np.isnan(arr)].tolist())
        all_raw = np.array(all_raw)

        if len(all_raw) < 2:
            ax_cdf.set_title(f"{d}\n(insufficient data)")
            continue

        x_min = all_raw.min() - 3
        x_max = all_raw.max() + 3
        x_grid = np.linspace(x_min, x_max, 600)

        vals, weights, kde_cdf, emp_cdf_tuple = build_weighted_cdf(
            d_det_vals, d_det_weights,
            d_ens_draws, d_ens_weights,
            x_grid,
        )

        cdf_store[d] = (x_grid, kde_cdf)

        # ── CDF panel ──────────────────────────────────────────────
        # Ensemble spread as shaded region
        for name, arr in d_ens_draws.items():
            clean = arr[~np.isnan(arr)]
            if len(clean) > 1:
                p10, p90 = np.percentile(clean, [10, 90])
                ax_cdf.axvspan(p10, p90, alpha=0.08,
                               color=ENS_MODELS[name]["color"], label=f"{name} P10-P90")

        # Deterministic model lines
        for name, v in d_det_vals.items():
            ax_cdf.axvline(v, color=DET_MODELS[name]["color"], alpha=0.7,
                           linewidth=1.5, linestyle="--", label=f"{name} ({v:.1f}°C)")

        # Empirical CDF
        if emp_cdf_tuple is not None:
            ax_cdf.step(emp_cdf_tuple[0], emp_cdf_tuple[1], where="post",
                        color="grey", linewidth=1.5, alpha=0.6, label="Empirical CDF")

        # KDE smoothed CDF
        ax_cdf.plot(x_grid, kde_cdf, color="black", linewidth=2.5, label="KDE CDF (weighted)")

        # Key quantiles
        for q, ls in [(10, ":"), (25, "-."), (50, "-"), (75, "-."), (90, ":")]:
            t_q = float(np.interp(q / 100, kde_cdf, x_grid))
            ax_cdf.axvline(t_q, color="steelblue", linewidth=0.8, linestyle=ls, alpha=0.5)
            ax_cdf.text(t_q, 0.02 + q / 500, f"Q{q}\n{t_q:.1f}°", fontsize=6,
                        color="steelblue", ha="center")

        # Threshold
        if args.threshold is not None:
            p_exc = float(1.0 - np.interp(args.threshold, x_grid, kde_cdf))
            ax_cdf.axvline(args.threshold, color="red", linewidth=2,
                           label=f"Threshold {args.threshold}°C  P(exc)={p_exc:.1%}")

        ax_cdf.set_title(f"{d}", fontsize=11, fontweight="bold")
        ax_cdf.set_xlabel("T_max (°C)", fontsize=9)
        ax_cdf.set_ylabel("CDF" if col_idx == 0 else "", fontsize=9)
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.grid(True, alpha=0.25)
        ax_cdf.legend(fontsize=6, loc="upper left", framealpha=0.8)

        # ── Box plot panel ─────────────────────────────────────────
        box_data = []
        box_labels = []
        box_colors = []

        for name, arr in d_ens_draws.items():
            clean = arr[~np.isnan(arr)]
            if len(clean) > 1:
                box_data.append(clean)
                box_labels.append(name.replace(" Ensemble", "").replace(" EPS", ""))
                box_colors.append(ENS_MODELS[name]["color"])

        if box_data:
            bp = ax_box.boxplot(box_data, vert=True, patch_artist=True,
                                medianprops=dict(color="black", linewidth=2),
                                whiskerprops=dict(linewidth=1.2),
                                flierprops=dict(markersize=3, alpha=0.4))
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            ax_box.set_xticklabels(box_labels, fontsize=7, rotation=15, ha="right")

        # Overlay deterministic model points
        for i_det, (name, v) in enumerate(d_det_vals.items()):
            ax_box.axhline(v, color=DET_MODELS[name]["color"], linewidth=1.2,
                           linestyle="--", alpha=0.8)

        ax_box.set_ylabel("T_max (°C)" if col_idx == 0 else "", fontsize=9)
        ax_box.grid(True, alpha=0.25, axis="y")
        ax_box.set_title("Ensemble spread", fontsize=8)

    n_det = len(det_series)
    n_ens_members = sum(df.shape[1] for df in ens_dfs.values())
    fig.suptitle(
        f"Probabilistic T_max Forecast — {args.city}\n"
        f"{n_det} deterministic models  +  {n_ens_members} ensemble members  "
        f"across {len(ens_dfs)} ensemble models",
        fontsize=11, fontweight="bold"
    )

    out_path = "forecast_cdf_ensemble.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")