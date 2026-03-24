from __future__ import annotations
import os
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from reflacxloader import ReflacxLoader
from stats_utils import finite, fit_gmm_2comp, compute_delta_bic
PARAMS = {"VALLEY_INTERVAL": (5.0, 40.0), "GMM_SUBSAMPLE_N": 500_000, "SAVGOL_WINDOW": 51, "SAVGOL_POLY": 3,"BINS": 150,"SEED": 123}
METRICS = ["turn_std", "velocity_std", "acceleration_std", "step_length_std"]
WINDOWS = [10, 20, 30, 40, 50]
def collect(loader: ReflacxLoader, metric: str, ws: int, stride: int) -> np.ndarray:
    xs: List[np.ndarray] = []
    for pid, sid in tqdm(loader.iter_study_pairs(max_studies=None), desc=f"{metric} ws={ws}", leave=False):
        try: xs.append(loader.get_study_window_features(pid, sid, ws, stride, use_cache=True)[metric])
        except Exception: continue
    return finite(np.concatenate(xs).astype(np.float64, copy=False))
def find_valley(x: np.ndarray, interval: Tuple[float, float], bins: int = PARAMS["BINS"]) -> float:
    density, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    wlen = min(PARAMS["SAVGOL_WINDOW"], len(density))
    if wlen % 2 == 0:
        wlen -= 1
    wlen = max(wlen, PARAMS["SAVGOL_POLY"] + 2)
    if wlen % 2 == 0:
        wlen -= 1
    smooth = savgol_filter(density, window_length=wlen, polyorder=PARAMS["SAVGOL_POLY"])
    low, high = interval
    idx_low = int(np.searchsorted(centers, low))
    idx_high = int(np.searchsorted(centers, high))
    idx_low = max(0, min(idx_low, len(centers) - 1))
    idx_high = max(idx_low + 1, min(idx_high, len(centers)))
    valley_idx = idx_low + int(np.argmin(smooth[idx_low:idx_high]))
    return float(centers[valley_idx])
C_TEAL = "#AFEEEE"
C_PLUM = "#DC92EF"
C_ROSE = "#c76076"

def make_plot(x: np.ndarray, metric: str, out_path: str, valley: float = None) -> None:
    p99_5 = float(np.percentile(x, 99.5))
    x = x[x <= p99_5]
    bins = 100
    density, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(density, sigma=3.0, mode="nearest")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(x, bins=edges, density=True, color=C_TEAL, edgecolor="#000000", linewidth=0.3, alpha=0.7)
    if valley is not None and np.isfinite(valley):
        ax.axvline(valley, color=C_ROSE, linewidth=2.0, linestyle="-", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
def analyze_metric_window(x: np.ndarray, metric: str, ws: int, stride: int, seed: int) -> Dict[str, object]:
    row: Dict[str, object] = {"metric": metric, "window_ms": int(ws), "stride_ms": int(stride), "n_windows": int(x.size)}
    p50 = float(np.percentile(x, 50))
    p99 = float(np.percentile(x, 99))
    row["p50"] = p50
    row["p99_over_p50"] = float(p99 / (p50 + 1e-12))
    valley = np.nan
    if metric == "turn_std": valley = find_valley(x, PARAMS["VALLEY_INTERVAL"])
    row["valley"] = float(valley) if np.isfinite(valley) else np.nan
    _, gmm_stats = fit_gmm_2comp( x, valley if metric == "turn_std" and np.isfinite(valley) else None, seed)
    row["pi_min"] = gmm_stats["pi_min"]
    row["cohen_d"] = gmm_stats["cohen_d"]
    row["delta_bic"] = compute_delta_bic( x, valley if metric == "turn_std" and np.isfinite(valley) else None, seed)
    return row, valley
def run_bimodality_analysis(loader: ReflacxLoader, out_csv: str, fig_dir: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    rows = []
    for metric in tqdm(METRICS, desc="metrics"):
        for ws in tqdm(WINDOWS, desc=f"windows ({metric})", leave=False):
            stride = ws // 2
            x = collect(loader, metric, ws, stride)
            row, valley = analyze_metric_window(x,metric,ws,stride,seed=PARAMS["SEED"] + ws + (hash(metric) & 0xFFFF))
            rows.append(row)
            make_plot(x, metric, out_path=os.path.join(fig_dir, f"{metric}_ws{ws}.pdf"), valley=valley)
    df = pd.DataFrame(rows).sort_values(["metric", "window_ms"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"Figures: {fig_dir}")

if __name__ == "__main__":
    OUT_CSV = "outputs/tables/reflacx_bimodality_quality.csv"
    FIG_DIR = "outputs/figures/reflacx_distributions"
    loader = ReflacxLoader()
    loader.load_jsons()
    run_bimodality_analysis(loader, OUT_CSV, FIG_DIR)