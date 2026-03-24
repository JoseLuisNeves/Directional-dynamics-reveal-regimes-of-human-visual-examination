from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from gazebuilder import GazeWindow
from gazebaseloader import GazeBaseLoader, SubjectSession
from stats_utils import mw_stats
from mode_intrep_reflacx import (
    directional_entropy,
    angular_reversal_density,
    curved_crossing,
    curved_return_rate,
    create_summary,
    INCLUDE_CURVED_METRICS
)

GAMES = ["Reading", "Video_1", "Balura_Game"]
WINDOWS = [20, 30, 40]
BASELINE_METRICS = ["turn_std", "velocity_std", "acceleration_std", "step_length_std"]

OUT_ABSOLUTE = "outputs/tables/complexity_gazebase_absolute.csv"
OUT_STRATIFIED = "outputs/tables/complexity_gazebase_stratified.csv"
OUT_SUMMARY = "outputs/tables/complexity_gazebase_summary.csv"
BIMODALITY_CSV = "outputs/tables/bimodality/gazebase_bimodality_quality.csv"

MAX_SUBJECTS: Optional[int] = None
CACHE_DIR = "outputs/cache/gazebase_complexity"


def _cache_path(game: str, ws: int, st: int, ss: SubjectSession, metric: str) -> str:
    d = os.path.join(CACHE_DIR, "complexity_proxies", metric, game, f"ws{ws}_st{st}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{ss.subject}__{ss.session}.npy")


def _load_or_compute(game: str, ws: int, st: int, ss: SubjectSession, windows: List[GazeWindow], use_cache: bool) -> Dict[str, np.ndarray]:
    # Always load/compute entropy and reversal
    cache_paths = {
        "dir_entropy": _cache_path(game, ws, st, ss, "dir_entropy"),
        "angular_reversal": _cache_path(game, ws, st, ss, "angular_reversal")
    }
    
    # Add curved metrics if enabled
    if INCLUDE_CURVED_METRICS:
        cache_paths["curved_crossing"] = _cache_path(game, ws, st, ss, "curved_crossing")
        cache_paths["curved_return"] = _cache_path(game, ws, st, ss, "curved_return")
    
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


def load_valleys(csv: str) -> Dict[Tuple[str, int], float]:
    if not os.path.exists(csv):
        return {}
    df = pd.read_csv(csv)
    df = df[df["metric"] == "turn_std"]
    out = {}
    for _, r in df.iterrows():
        if pd.notna(r.get("valley")) and pd.notna(r.get("window_ms")) and pd.notna(r.get("game")):
            out[(str(r["game"]), int(r["window_ms"]))] = float(r["valley"])
    return out


def collect(loader: GazeBaseLoader, game: str, ws: int, st: int, max_subjects: Optional[int]) -> Dict[str, Any]:
    metrics = BASELINE_METRICS + ["spatial_dispersion", "dir_entropy", "angular_reversal"]
    if INCLUDE_CURVED_METRICS:
        metrics += ["curved_crossing", "curved_return"]
    
    acc: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
    n_subjects = 0
    
    for ss in tqdm(loader.iter_subject_sessions(game, [1], max_subjects=max_subjects), desc=f"{game} ws={ws}", leave=False):
        try:
            feats = loader.get_subject_window_features(ss, ws, st, use_cache=True)
            windows = loader.get_subject_windows(ss, ws, st)
            n = min(len(windows), len(feats["turn_std"]), 
                   *(len(feats.get(m, [])) for m in BASELINE_METRICS + ["spatial_dispersion"]))
            if n < 10:
                continue
            
            prox = _load_or_compute(game, ws, st, ss, windows[:n], use_cache=True)
            
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
            
            n_subjects += 1
        except:
            continue
    
    out = {m: (np.concatenate(acc[m]) if acc[m] else np.array([], np.float32)) for m in metrics}
    out["n_subjects"] = n_subjects
    return out


def analyze_absolute(data: Dict[str, np.ndarray], game: str, ws: int, turn_valley: Optional[float]) -> List[Dict]:
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
                "game": game,
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


def analyze_stratified(data: Dict[str, np.ndarray], game: str, ws: int, turn_valley: Optional[float]) -> List[Dict]:
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
                    "game": game,
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


def main():
    loader = GazeBaseLoader()
    valley_map = load_valleys(BIMODALITY_CSV)
    
    absolute_results = []
    stratified_results = []
    
    for game in GAMES:
        for ws in WINDOWS:
            st = ws // 2
            print(f"\nProcessing {game} {ws}ms...")
            data = collect(loader, game, ws, st, MAX_SUBJECTS)
            valley = valley_map.get((game, ws))
            print(f"  Absolute analysis...")
            absolute_results.extend(analyze_absolute(data, game, ws, valley))
            print(f"  Stratified analysis...")
            stratified_results.extend(analyze_stratified(data, game, ws, valley))
    
    absolute_df = pd.DataFrame(absolute_results)
    stratified_df = pd.DataFrame(stratified_results)
    
    os.makedirs(os.path.dirname(OUT_ABSOLUTE), exist_ok=True)
    absolute_df.to_csv(OUT_ABSOLUTE, index=False)
    stratified_df.to_csv(OUT_STRATIFIED, index=False)
    print(f"\nSaved: {OUT_ABSOLUTE}")
    print(f"Saved: {OUT_STRATIFIED}")
    
    summary_df = create_summary(absolute_df, stratified_df)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    print(f"Saved: {OUT_SUMMARY}")
    print("\nDone!")


if __name__ == "__main__":
    main()