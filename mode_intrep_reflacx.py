from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from gazebuilder import GazeWindow
from reflacxloader import ReflacxLoader
from stats_utils import mw_stats

WINDOWS = [20, 30, 40]
BASELINE_METRICS = ["turn_std", "velocity_std", "acceleration_std", "step_length_std"]
INCLUDE_CURVED_METRICS = True

OUT_ABSOLUTE = "outputs/tables/complexity_reflacx_absolute.csv"
OUT_STRATIFIED = "outputs/tables/complexity_reflacx_stratified.csv"
OUT_SUMMARY = "outputs/tables/complexity_reflacx_summary.csv"
BIMODALITY_CSV = "outputs/tables/bimodality/reflacx_bimodality_quality.csv"
FIG_HEATMAP = "outputs/figures/reflacx_complexity/heatmap_absolute.pdf"
FIG_STRATIFIED = "outputs/figures/reflacx_complexity/stratified_trends.pdf"
FIG_COMPARISON = "outputs/figures/reflacx_complexity/q1_vs_q4.pdf"

MAX_STUDIES: Optional[int] = None
CACHE_DIR = "outputs/cache/reflacx_complexity"

C_TEAL = "#AFEEEE"
C_PLUM = "#DC92EF"
C_ROSE = "#c76076"


def directional_entropy(coords: np.ndarray, n_bins: int = 8) -> float:
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(angles, bins=bin_edges)
    probs = counts / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def angular_reversal_density(coords: np.ndarray) -> float:
    if len(coords) < 3:
        return 0.0
    v1 = coords[1:-1] - coords[:-2]
    v2 = coords[2:] - coords[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-6) & (n2 > 1e-6)
    if not np.any(valid):
        return 0.0
    v1 = v1[valid]
    v2 = v2[valid]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = np.sum(v1 * v2, axis=1)
    angles = np.degrees(np.arctan2(cross, dot))
    signs = np.sign(angles)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    flips = np.sum(signs[1:] != signs[:-1])
    return float(flips / len(signs))


def segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """Check if line segment p1-p2 intersects with line segment p3-p4."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def curved_crossing(coords: np.ndarray, min_turn_deg: float = 180.0) -> float:
    """
    Count self-intersections that are supported by sufficient cumulative turning
    along the subpath between intersecting segments.

    Returns crossings normalized by path length.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) < 4:
        return 0.0

    n_segs = len(coords) - 1
    crossings = 0

    for i in range(n_segs - 2):
        for j in range(i + 2, n_segs):
            if not segments_intersect(coords[i], coords[i + 1], coords[j], coords[j + 1]):
                continue

            path = coords[i : j + 2]
            if len(path) < 4:
                continue

            v1 = path[1:-1] - path[:-2]
            v2 = path[2:] - path[1:-1]
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            valid = (n1 > 1e-6) & (n2 > 1e-6)
            if not np.any(valid):
                continue

            cos_ang = np.sum(v1[valid] * v2[valid], axis=1) / (n1[valid] * n2[valid] + 1e-12)
            angles = np.abs(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

            if float(np.sum(angles)) >= float(min_turn_deg):
                crossings += 1

    path_len = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
    return float(crossings / (path_len + 1e-6))


def curved_return_rate(coords, min_turn_deg=180, eps_percentile=20, min_lag=2):
    """
    Counts returns to a prior location (within eps radius) that are supported by
    sufficient cumulative turning along the path between the first visit and return.

    Returns curved_returns / n_points.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) <= min_lag:
        return 0.0

    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    eps = np.percentile(steps, eps_percentile) if len(steps) > 0 else 1.0
    if eps <= 0:
        return 0.0

    eps2 = float(eps ** 2)
    n = len(coords)
    curved_returns = 0

    for i in range(min_lag, n):
        d2 = np.sum((coords[: i - min_lag] - coords[i]) ** 2, axis=1)
        return_indices = np.where(d2 <= eps2)[0]
        if len(return_indices) == 0:
            continue

        j = int(return_indices[0])
        if i - j < 2:
            continue

        path = coords[j : i + 1]
        if len(path) < 3:
            continue

        v1 = path[1:-1] - path[:-2]
        v2 = path[2:] - path[1:-1]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        valid = (n1 > 1e-6) & (n2 > 1e-6)
        if not np.any(valid):
            continue

        cos_angles = np.sum(v1[valid] * v2[valid], axis=1) / (n1[valid] * n2[valid] + 1e-12)
        angles = np.abs(np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0))))
        cumulative_turn = float(np.sum(angles))

        if cumulative_turn >= float(min_turn_deg):
            curved_returns += 1

    return float(curved_returns) / float(n)


def _cache_path(pid: str, sid: str, ws: int, st: int, metric: str) -> str:
    d = os.path.join(CACHE_DIR, "complexity_proxies", metric, f"ws{ws}_st{st}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{pid}__{sid}.npy")


def _load_or_compute(pid: str, sid: str, ws: int, st: int, windows: List[GazeWindow], use_cache: bool) -> Dict[str, np.ndarray]:
    # Always load/compute entropy and reversal
    cache_paths = {
        "dir_entropy": _cache_path(pid, sid, ws, st, "dir_entropy"),
        "angular_reversal": _cache_path(pid, sid, ws, st, "angular_reversal")
    }
    
    # Add curved metrics if enabled
    if INCLUDE_CURVED_METRICS:
        cache_paths["curved_crossing"] = _cache_path(pid, sid, ws, st, "curved_crossing")
        cache_paths["curved_return"] = _cache_path(pid, sid, ws, st, "curved_return")
    
    # Check if all cached
    if use_cache and all(os.path.exists(p) for p in cache_paths.values()):
        return {k: np.load(p) for k, p in cache_paths.items()}
    
    # Compute metrics
    dir_ent, ang_rev = [], []
    curved_cross, curved_ret = [], []
    
    for w in windows:
        coords = np.asarray(w.coords, np.float64)
        dir_ent.append(directional_entropy(coords))
        ang_rev.append(angular_reversal_density(coords))
        
        if INCLUDE_CURVED_METRICS:
            curved_cross.append(curved_crossing(coords))
            curved_ret.append(curved_return_rate(coords))
    
    results = {
        "dir_entropy": np.array(dir_ent, np.float32),
        "angular_reversal": np.array(ang_rev, np.float32)
    }
    
    if INCLUDE_CURVED_METRICS:
        results["curved_crossing"] = np.array(curved_cross, np.float32)
        results["curved_return"] = np.array(curved_ret, np.float32)
    
    # Save to cache
    if use_cache:
        for k, arr in results.items():
            np.save(cache_paths[k], arr)
    
    return results


def load_valleys(csv: str) -> Dict[int, float]:
    if not os.path.exists(csv):
        return {}
    df = pd.read_csv(csv)
    df = df[df["metric"] == "turn_std"]
    out: Dict[int, float] = {}
    for _, r in df.iterrows():
        if pd.notna(r.get("valley")) and pd.notna(r.get("window_ms")):
            out[int(r["window_ms"])] = float(r["valley"])
    return out


def collect(loader: ReflacxLoader, ws: int, st: int, max_studies: Optional[int]) -> Dict[str, Any]:
    metrics = BASELINE_METRICS + ["spatial_dispersion", "dir_entropy", "angular_reversal"]
    if INCLUDE_CURVED_METRICS:
        metrics += ["curved_crossing", "curved_return"]
    
    acc: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
    n_studies = 0
    study_pairs = loader.iter_study_pairs(max_studies)
    total = max_studies if max_studies is not None else None
    
    for pid, sid in tqdm(study_pairs, desc=f"ws={ws}", leave=False, total=total):
        try:
            feats = loader.get_study_window_features(pid, sid, ws, st, use_cache=True)
            windows = loader.get_study_windows(pid, sid, ws, st, filter_to_chest=True)
            n = min(len(windows), len(feats["turn_std"]), 
                   *(len(feats.get(m, [])) for m in BASELINE_METRICS + ["spatial_dispersion"]))
            if n < 10:
                continue
            
            prox = _load_or_compute(pid, sid, ws, st, windows[:n], use_cache=True)
            
            valid = np.ones(n, dtype=bool)
            for m in BASELINE_METRICS + ["spatial_dispersion"]:
                valid &= np.isfinite(feats[m][:n])
            for k in prox:
                valid &= np.isfinite(prox[k][:n])
            
            if not np.any(valid):
                continue
            
            for m in BASELINE_METRICS + ["spatial_dispersion"]:
                acc[m].append(np.asarray(feats[m][:n][valid], np.float32))
            for k in prox:
                acc[k].append(np.asarray(prox[k][:n][valid], np.float32))
            
            n_studies += 1
        except:
            continue
    
    out = {m: (np.concatenate(acc[m]) if acc[m] else np.array([], np.float32)) for m in metrics}
    out["n_studies"] = n_studies
    return out


def analyze_absolute(data: Dict[str, np.ndarray], ws: int, turn_valley: Optional[float]) -> List[Dict]:
    outcomes = ["dir_entropy", "angular_reversal"]
    if INCLUDE_CURVED_METRICS:
        outcomes += ["curved_crossing", "curved_return"]
    outcomes += ["spatial_dispersion"]
    
    required = BASELINE_METRICS + outcomes
    n_min = min(len(data[m]) for m in required)
    valid = np.ones(n_min, dtype=bool)
    for m in required:
        valid &= np.isfinite(data[m][:n_min])
    
    turn = data["turn_std"][:n_min][valid]
    vel = data["velocity_std"][:n_min][valid]
    acc = data["acceleration_std"][:n_min][valid]
    step = data["step_length_std"][:n_min][valid]
    
    splits = {
        "turn_valley": turn <= turn_valley if turn_valley is not None else None,
        "turn_med": turn <= np.median(turn),
        "vel_med": vel <= np.median(vel),
        "acc_med": acc <= np.median(acc),
        "step_med": step <= np.median(step)
    }
    
    results = []
    for outcome in outcomes:
        y = data[outcome][:n_min][valid]
        for split_name, low_mask in splits.items():
            if low_mask is None:
                continue
            high_mask = ~low_mask
            stats = mw_stats(y[low_mask], y[high_mask])
            mean_diff = y[high_mask].mean() - y[low_mask].mean()
            pooled_std = np.sqrt((y[high_mask].std()**2 + y[low_mask].std()**2) / 2)
            cohens_d = mean_diff / (pooled_std + 1e-9)
            results.append({
                "window_ms": ws,
                "quartile": "All",
                "outcome": outcome,
                "splitter": split_name,
                "n_low": int(low_mask.sum()),
                "n_high": int(high_mask.sum()),
                "mean_low": float(y[low_mask].mean()),
                "mean_high": float(y[high_mask].mean()),
                "mean_diff": float(mean_diff),
                "cohens_d": float(cohens_d),
                "rbc": float(stats.get("rbc", np.nan)),
                "p_value": float(stats.get("p", np.nan))
            })
    return results


def analyze_stratified(data: Dict[str, np.ndarray], ws: int, turn_valley: Optional[float]) -> List[Dict]:
    outcomes = ["dir_entropy", "angular_reversal"]
    if INCLUDE_CURVED_METRICS:
        outcomes += ["curved_crossing", "curved_return"]
    
    required = BASELINE_METRICS + ["spatial_dispersion"] + outcomes
    n_min = min(len(data[m]) for m in required)
    valid = np.ones(n_min, dtype=bool)
    for m in required:
        valid &= np.isfinite(data[m][:n_min])
    
    turn = data["turn_std"][:n_min][valid]
    vel = data["velocity_std"][:n_min][valid]
    acc = data["acceleration_std"][:n_min][valid]
    step = data["step_length_std"][:n_min][valid]
    disp = data["spatial_dispersion"][:n_min][valid]
    
    q_edges = np.percentile(disp, [0, 25, 50, 75, 100])
    quartiles = np.digitize(disp, q_edges[:-1]) - 1
    quartiles = np.clip(quartiles, 0, 3)
    
    results = []
    for q_idx, q_name in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        q_mask = quartiles == q_idx
        if q_mask.sum() < 20:
            continue
        
        turn_q = turn[q_mask]
        vel_q = vel[q_mask]
        acc_q = acc[q_mask]
        step_q = step[q_mask]
        
        splits = {
            "turn_valley": turn_q <= turn_valley if turn_valley is not None else None,
            "turn_med": turn_q <= np.median(turn_q),
            "vel_med": vel_q <= np.median(vel_q),
            "acc_med": acc_q <= np.median(acc_q),
            "step_med": step_q <= np.median(step_q)
        }     
        
        for outcome in outcomes:
            y_q = data[outcome][:n_min][valid][q_mask]
            for split_name, low_mask in splits.items():
                if low_mask is None:
                    continue
                high_mask = ~low_mask
                if low_mask.sum() < 5 or high_mask.sum() < 5:
                    continue
                stats = mw_stats(y_q[low_mask], y_q[high_mask])
                mean_diff = y_q[high_mask].mean() - y_q[low_mask].mean()
                pooled_std = np.sqrt((y_q[high_mask].std()**2 + y_q[low_mask].std()**2) / 2)
                cohens_d = mean_diff / (pooled_std + 1e-9)
                results.append({
                    "window_ms": ws,
                    "quartile": q_name,
                    "outcome": outcome,
                    "splitter": split_name,
                    "n_low": int(low_mask.sum()),
                    "n_high": int(high_mask.sum()),
                    "mean_low": float(y_q[low_mask].mean()),
                    "mean_high": float(y_q[high_mask].mean()),
                    "mean_diff": float(mean_diff),
                    "cohens_d": float(cohens_d),
                    "rbc": float(stats.get("rbc", np.nan)),
                    "p_value": float(stats.get("p", np.nan))
                })
    return results


def create_summary(absolute_df: pd.DataFrame, stratified_df: pd.DataFrame) -> pd.DataFrame:
    outcomes = ["dir_entropy", "angular_reversal"]
    if INCLUDE_CURVED_METRICS:
        outcomes += ["curved_crossing", "curved_return"]
    
    rows = []
    
    # Absolute analysis - include spatial_dispersion
    for ws in WINDOWS:
        subset = absolute_df[absolute_df["window_ms"] == ws]
        for outcome in outcomes + ["spatial_dispersion"]:
            for splitter in ["turn_valley", "vel_med", "acc_med", "step_med"]:
                vals = subset[(subset["outcome"] == outcome) & (subset["splitter"] == splitter)]
                if len(vals) > 0:
                    rows.append({
                        "window_ms": ws,
                        "analysis": "absolute",
                        "outcome": outcome,
                        "splitter": splitter,
                        "mean_rbc": vals["rbc"].abs().mean(),
                        "mean_cohens_d": vals["cohens_d"].abs().mean()
                    })
    
    # Stratified analysis - exclude spatial_dispersion
    for ws in WINDOWS:
        subset = stratified_df[(stratified_df["window_ms"] == ws) & (stratified_df["quartile"] == "Q1")]
        for outcome in outcomes:
            for splitter in ["turn_valley", "vel_med", "acc_med", "step_med"]:
                vals = subset[(subset["outcome"] == outcome) & (subset["splitter"] == splitter)]
                if len(vals) > 0:
                    rows.append({
                        "window_ms": ws,
                        "analysis": "Q1_stratified",
                        "outcome": outcome,
                        "splitter": splitter,
                        "mean_rbc": vals["rbc"].abs().mean(),
                        "mean_cohens_d": vals["cohens_d"].abs().mean()
                    })
    
    return pd.DataFrame(rows)


def main():
    loader = ReflacxLoader()
    loader.load_jsons()
    valley_map = load_valleys(BIMODALITY_CSV)
    
    absolute_results = []
    stratified_results = []
    
    for ws in WINDOWS:
        st = ws // 2
        data = collect(loader, ws, st, MAX_STUDIES)
        valley = valley_map.get(ws)
        absolute_results.extend(analyze_absolute(data, ws, valley))
        stratified_results.extend(analyze_stratified(data, ws, valley))
    
    absolute_df = pd.DataFrame(absolute_results)
    stratified_df = pd.DataFrame(stratified_results)
    
    os.makedirs(os.path.dirname(OUT_ABSOLUTE), exist_ok=True)
    absolute_df.to_csv(OUT_ABSOLUTE, index=False)
    stratified_df.to_csv(OUT_STRATIFIED, index=False)
    
    summary_df = create_summary(absolute_df, stratified_df)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()