import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from scipy.stats import spearmanr, false_discovery_control
from reflacxloader import ReflacxLoader

# Config
WINDOW_CONFIGS = [
    {"window_size": 30, "stride": 15},
    {"window_size": 20, "stride": 10},
    {"window_size": 10, "stride": 5}
]
MAX_STUDIES = None
N_BOOTSTRAP = 1000
OUT_SUMMARY = "outputs/tables/unified_validation_multiwindow.csv"
REGIMES = ["HiF_LoV", "HiF_HiV", "LoF_LoV", "LoF_HiV"]


def regime_from_flags(f_high: bool, v_high: bool) -> str:
    if f_high and not v_high:
        return "HiF_LoV"
    if f_high and v_high:
        return "HiF_HiV"
    if not f_high and not v_high:
        return "LoF_LoV"
    return "LoF_HiV"


def compute_topology_metrics(windows) -> dict:
    """Compute topology metrics on 10×10 grid."""
    if len(windows) < 10:
        return {k: np.nan for k in ["transition_entropy", "unique_cells", "self_loops", 
                                     "stationary_gaze_entropy", "determinism"]}
    
    centers = np.array([w.center for w in windows])
    x_min, x_max = centers[:, 0].min(), centers[:, 0].max()
    y_min, y_max = centers[:, 1].min(), centers[:, 1].max()
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    
    grid_size = 10
    x_bins = ((centers[:, 0] - x_min) / x_range * (grid_size - 1)).astype(int)
    y_bins = ((centers[:, 1] - y_min) / y_range * (grid_size - 1)).astype(int)
    cells = x_bins * grid_size + y_bins
    
    unique_cells = len(np.unique(cells))
    
    # SGE
    cell_counts = np.bincount(cells, minlength=grid_size**2)
    p_sge = cell_counts / cell_counts.sum()
    p_sge = p_sge[p_sge > 0]
    sge = -np.sum(p_sge * np.log2(p_sge))
    
    # Transitions
    transitions = list(zip(cells[:-1], cells[1:]))
    if len(transitions) == 0:
        return {"transition_entropy": np.nan, "unique_cells": unique_cells, 
                "self_loops": np.nan, "stationary_gaze_entropy": float(sge), 
                "determinism": np.nan}
    
    self_loops = sum(1 for a, b in transitions if a == b) / len(transitions)
    
    from collections import Counter
    trans_counts = Counter(transitions)
    probs = np.array([c / sum(trans_counts.values()) for c in trans_counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    # RQA Determinism
    n = len(cells)
    recurrence = np.array([[cells[i] == cells[j] for j in range(n)] for i in range(n)])
    diag_points = 0
    for offset in range(-(n-1), n):
        diag = np.diagonal(recurrence, offset=offset)
        if len(diag) < 2:
            continue
        current_len = 0
        for val in diag:
            if val:
                current_len += 1
            elif current_len >= 2:
                diag_points += current_len
                current_len = 0
        if current_len >= 2:
            diag_points += current_len
    determinism = diag_points / max(1, np.sum(recurrence))
    
    return {
        "transition_entropy": float(entropy),
        "unique_cells": int(unique_cells),
        "self_loops": float(self_loops),
        "stationary_gaze_entropy": float(sge),
        "determinism": float(determinism)
    }


def fit_nested_models(df, outcome):
    """Fit M1→M4 and return R² progression."""
    regime_cols = [f"p_{r}" for r in REGIMES[:-1]]  # Omit LoF_HiV reference
    needed = [outcome, "mean_velocity", "velocity_std", "mean_acceleration", 
              "acceleration_std", "mean_turn_angle"] + regime_cols
    
    dfm = df[needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(dfm) < 50:
        return None
    
    y = dfm[outcome].to_numpy()
    X1 = dfm[["mean_velocity"]].to_numpy()
    X2 = dfm[["mean_velocity", "velocity_std", "mean_acceleration", "acceleration_std"]].to_numpy()
    X3 = dfm[["mean_velocity", "velocity_std", "mean_acceleration", "acceleration_std", 
              "mean_turn_angle"]].to_numpy()
    X4 = dfm[list(dfm.columns[1:])].to_numpy()  # All predictors
    
    m1 = sm.OLS(y, sm.add_constant(X1)).fit()
    m2 = sm.OLS(y, sm.add_constant(X2)).fit()
    m3 = sm.OLS(y, sm.add_constant(X3)).fit()
    m4 = sm.OLS(y, sm.add_constant(X4)).fit()
    
    try:
        p_incr = float(anova_lm(m3, m4).iloc[1]["Pr(>F)"])
    except:
        p_incr = np.nan
    
    rho_vel, p_vel = spearmanr(dfm["mean_velocity"], dfm[outcome])
    rho_hif, p_hif = spearmanr(dfm[regime_cols[0]], dfm[outcome])
    
    return {
        "n": len(dfm),
        "r2_m1": m1.rsquared,
        "r2_m2": m2.rsquared,
        "r2_m3": m3.rsquared,
        "r2_m4": m4.rsquared,
        "delta_r2": m4.rsquared - m3.rsquared,
        "p_incremental": p_incr,
        "rho_velocity": rho_vel,
        "p_velocity": p_vel,
        "rho_HiF_LoV": rho_hif,
        "p_HiF_LoV": p_hif
    }


def bootstrap_delta_r2(df, outcome, n_boot=1000):
    """Bootstrap CI for ΔR² from M3→M4."""
    regime_cols = [f"p_{r}" for r in REGIMES[:-1]]
    needed = [outcome, "mean_velocity", "velocity_std", "mean_acceleration", 
              "acceleration_std", "mean_turn_angle"] + regime_cols
    
    dfm = df[needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(dfm) < 50:
        return {"boot_ci_lower": np.nan, "boot_ci_upper": np.nan}
    
    deltas = []
    for i in range(n_boot):
        boot_df = dfm.sample(n=len(dfm), replace=True, random_state=i)
        y = boot_df[outcome].to_numpy()
        
        X3 = boot_df[["mean_velocity", "velocity_std", "mean_acceleration", 
                      "acceleration_std", "mean_turn_angle"]].to_numpy()
        X4 = boot_df[list(boot_df.columns[1:])].to_numpy()
        
        try:
            m3 = sm.OLS(y, sm.add_constant(X3)).fit()
            m4 = sm.OLS(y, sm.add_constant(X4)).fit()
            deltas.append(m4.rsquared - m3.rsquared)
        except:
            continue
    
    if len(deltas) < 100:
        return {"boot_ci_lower": np.nan, "boot_ci_upper": np.nan}
    
    return {
        "boot_ci_lower": np.percentile(deltas, 2.5),
        "boot_ci_upper": np.percentile(deltas, 97.5)
    }


def cohens_f2(r2_full, r2_reduced):
    """Cohen's f² effect size."""
    if r2_full >= 1.0:
        return np.nan
    return (r2_full - r2_reduced) / (1 - r2_full)


def compute_medians(loader, window_size, stride, n_sample=300):
    """Compute dataset-wide medians for turn_std and velocity_std."""
    print(f"Computing medians for ws={window_size}, stride={stride}...")
    all_turn, all_vel = [], []
    
    for pid, sid in tqdm(list(loader.iter_study_pairs(n_sample)), desc="medians"):
        try:
            feats = loader.get_study_window_features(pid, sid, window_size, stride, 
                                                     period="pre-reporting", use_cache=True)
            turn = feats.get("turn_std", np.array([]))
            vel = feats.get("velocity_std", np.array([]))
            turn = turn[np.isfinite(turn)]
            vel = vel[np.isfinite(vel)]
            if len(turn) > 0:
                all_turn.append(turn)
            if len(vel) > 0:
                all_vel.append(vel)
        except:
            continue
    
    turn_median = float(np.median(np.concatenate(all_turn)))
    vel_median = float(np.median(np.concatenate(all_vel)))
    print(f"  Turn median: {turn_median:.3f}, Velocity median: {vel_median:.1f}")
    return turn_median, vel_median


def analyze_window_config(loader, window_size, stride):
    """Analyze all studies for a given window configuration."""
    # Compute thresholds
    turn_median, vel_median = compute_medians(loader, window_size, stride)
    
    # Collect study-level data
    print(f"\nAnalyzing studies (ws={window_size}, stride={stride})...")
    rows = []
    
    for pid, sid in tqdm(list(loader.iter_study_pairs(MAX_STUDIES)), desc="studies"):
        try:
            windows = loader.get_study_windows(pid, sid, window_size, stride, 
                                              filter_to_chest=True, period="pre-reporting")
            feats = loader.get_study_window_features(pid, sid, window_size, stride, 
                                                     period="pre-reporting", use_cache=True)
            
            if len(windows) < 10:
                continue
            
            # Time to report
            word_ts = loader.word_timestamps_dict.get(pid, {}).get(sid)
            if not word_ts:
                continue
            
            reporting_start = float(word_ts[0]["timestamp_start_word"])
            first_gaze = float(windows[0].timestamps[0])
            time_to_report = reporting_start - first_gaze
            
            if not (5.0 <= time_to_report <= 300.0):
                continue
            
            # Window features
            turn_std = feats.get("turn_std", np.array([]))
            vel_std = feats.get("velocity_std", np.array([]))
            acc_std = feats.get("acceleration_std", np.array([]))
            
            # Compute means from windows directly
            turn_angles_all = []
            vel_all = []
            acc_all = []
            for w in windows:
                ta = w.turn_angles
                v = w.velocity
                a = w.acceleration
                turn_angles_all.extend(ta[np.isfinite(ta)])
                vel_all.extend(v[np.isfinite(v)])
                acc_all.extend(a[np.isfinite(a)])
            
            mean_turn_angle = float(np.mean(turn_angles_all)) if len(turn_angles_all) > 0 else np.nan
            mean_velocity = float(np.mean(vel_all)) if len(vel_all) > 0 else np.nan
            mean_acceleration = float(np.mean(acc_all)) if len(acc_all) > 0 else np.nan
            
            # Valid windows
            n = min(len(turn_std), len(vel_std))
            if n < 10:
                continue
            
            turn_std = turn_std[:n]
            vel_std = vel_std[:n]
            acc_std = acc_std[:n]
            
            valid = np.isfinite(turn_std) & np.isfinite(vel_std)
            if valid.sum() < 10:
                continue
            
            turn_std = turn_std[valid]
            vel_std = vel_std[valid]
            acc_std = acc_std[valid]
            
            # Regime classification: median split
            f_high = turn_std > turn_median
            v_high = vel_std > vel_median
            regimes = [regime_from_flags(bool(f_high[i]), bool(v_high[i])) 
                      for i in range(len(turn_std))]
            
            counts = {r: sum(1 for x in regimes if x == r) for r in REGIMES}
            total = len(regimes)
            
            # Topology metrics
            topo = compute_topology_metrics(windows)
            
            # Store
            row = {
                "pid": pid,
                "sid": sid,
                "window_size": window_size,
                "stride": stride,
                "time_to_report": time_to_report,
                "transition_entropy": topo["transition_entropy"],
                "unique_cells": topo["unique_cells"],
                "self_loops": topo["self_loops"],
                "stationary_gaze_entropy": topo["stationary_gaze_entropy"],
                "determinism": topo["determinism"],
                "mean_velocity": mean_velocity,
                "mean_acceleration": mean_acceleration,
                "velocity_std": float(np.mean(vel_std)),
                "acceleration_std": float(np.mean(acc_std)),
                "mean_turn_angle": mean_turn_angle,
                "n_windows": total
            }
            
            for r in REGIMES:
                row[f"p_{r}"] = counts[r] / total
            
            rows.append(row)
            
        except Exception as e:
            continue
    
    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} studies")
    return df


def main():
    os.makedirs("outputs/tables", exist_ok=True)
    
    loader = ReflacxLoader()
    loader.load_jsons()
    
    # Analyze each window configuration
    all_results = []
    
    for config in WINDOW_CONFIGS:
        ws = config["window_size"]
        stride = config["stride"]
        
        print(f"\n{'='*80}")
        print(f"WINDOW SIZE: {ws}ms, STRIDE: {stride}ms")
        print(f"{'='*80}")
        
        df = analyze_window_config(loader, ws, stride)
        
        if len(df) < 50:
            print(f"Insufficient data for ws={ws}, stride={stride}")
            continue
        
        # Analyze outcomes
        outcomes = ["transition_entropy", "unique_cells", "self_loops", 
                    "stationary_gaze_entropy", "determinism", "time_to_report"]
        
        for outcome in outcomes:
            print(f"\n  Analyzing {outcome}...")
            
            res = fit_nested_models(df, outcome)
            if not res:
                continue
            
            boot = bootstrap_delta_r2(df, outcome, N_BOOTSTRAP)
            f2 = cohens_f2(res["r2_m4"], res["r2_m3"])
            
            all_results.append({
                "window_size": ws,
                "stride": stride,
                "outcome": outcome,
                **res,
                **boot,
                "cohens_f2": f2
            })
    
    if not all_results:
        print("No results to save")
        return
    
    summary = pd.DataFrame(all_results)
    
    # FDR correction within each outcome
    for outcome in summary["outcome"].unique():
        mask = summary["outcome"] == outcome
        p_values = summary.loc[mask, "p_incremental"].values
        summary.loc[mask, "p_incremental_adj"] = false_discovery_control(p_values, method='bh')
    
    # Effect size interpretation
    summary["effect_size"] = pd.cut(
        summary["cohens_f2"],
        bins=[-np.inf, 0.02, 0.15, 0.35, np.inf],
        labels=["negligible", "small", "medium", "large"]
    )
    
    # Format and save
    summary = summary[[
        "window_size", "stride", "outcome", "n", "r2_m1", "r2_m2", "r2_m3", "r2_m4", 
        "delta_r2", "cohens_f2", "effect_size", "p_incremental", "p_incremental_adj",
        "boot_ci_lower", "boot_ci_upper", "rho_velocity", "p_velocity", 
        "rho_HiF_LoV", "p_HiF_LoV"
    ]]
    
    summary.to_csv(OUT_SUMMARY, index=False, float_format="%.6f")
    print(f"\n✓ Saved: {OUT_SUMMARY}\n")
    print(summary.to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY BY WINDOW SIZE")
    print("="*80)
    
    for ws in sorted(summary["window_size"].unique()):
        sub = summary[summary["window_size"] == ws]
        sig = sub[sub["p_incremental_adj"] < 0.05]
        stride_val = sub["stride"].iloc[0]
        
        print(f"\n{ws}ms window (stride={stride_val}ms): {len(sig)}/{len(sub)} outcomes significant (FDR < 0.05)")
        print(f"  ΔR² range: [{sub['delta_r2'].min():.3f}, {sub['delta_r2'].max():.3f}]")
        print(f"  f² range: [{sub['cohens_f2'].min():.3f}, {sub['cohens_f2'].max():.3f}]")
        print(f"  Effect sizes: {sub['effect_size'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
