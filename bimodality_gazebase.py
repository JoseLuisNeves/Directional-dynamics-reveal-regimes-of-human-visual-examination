from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from bimodality_reflacx import find_valley, make_plot
from gazebaseloader import GazeBaseLoader
from stats_utils import fit_gmm_2comp, compute_delta_bic


PARAMS = {"VALLEY_INTERVAL": (5.0, 40.0), "BINS": 150, "SEED": 123}

METRICS = ["turn_std", "velocity_std", "acceleration_std", "step_length_std"]
WINDOWS = [10, 20, 30, 40, 50]

DEFAULT_GAMES = ["Reading", "Video_1", "Balura_Game"]
DEFAULT_ROUNDS_BY_GAME = {"Reading": [1], "Video_1": [1], "Balura_Game": [1]}

OUT_CSV = "outputs/tables/gazebase_bimodality_quality.csv"
FIG_DIR = "outputs/figures/gazebase_distributions"

MAX_SUBJECTS: Optional[int] = None
MAX_WINDOWS_PER_SUBJECT = 50_000


def subject_variance_ratio_from_per_subject(pooled_x: np.ndarray, per_subject: Dict[str, np.ndarray]) -> Tuple[float, int]:
    """
    Subject Variance Ratio (SVR)
    SVR = Var( subject_means ) / Var( pooled )
    Uses unweighted variance over subject means.
    """
    x = np.asarray(pooled_x, np.float64)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return (np.nan, 0)

    means: List[float] = []
    for _, arr in per_subject.items():
        a = np.asarray(arr, np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        means.append(float(np.mean(a)))

    if len(means) < 3:
        return (np.nan, len(means))

    var_total = float(np.var(x))
    if not np.isfinite(var_total) or var_total <= 0.0:
        return (np.nan, len(means))

    var_between = float(np.var(np.asarray(means, np.float64)))
    svr = float(var_between / var_total)
    return (svr, len(means))


def analyze_metric_window(
    x: np.ndarray,
    game: str,
    rounds: List[int],
    metric: str,
    ws: int,
    stride: int,
    seed: int,
    n_subjects: int,
    subject_variance_ratio: float,
    n_subjects_for_svr: int,
) -> Tuple[Dict[str, object], float]:
    row: Dict[str, object] = {
        "dataset": "gazebase",
        "game": game,
        "rounds": ",".join(map(str, rounds)),
        "metric": metric,
        "window_ms": int(ws),
        "stride_ms": int(stride),
        "n_windows": int(x.size),
        "n_subjects": int(n_subjects),
        "subject_variance_ratio": float(subject_variance_ratio) if np.isfinite(subject_variance_ratio) else np.nan,
        "n_subjects_for_svr": int(n_subjects_for_svr),
    }

    p50 = float(np.percentile(x, 50))
    p99 = float(np.percentile(x, 99))
    row["p50"] = p50
    row["p99_over_p50"] = float(p99 / (p50 + 1e-12))

    valley = np.nan
    if metric == "turn_std":
        valley = find_valley(x, PARAMS["VALLEY_INTERVAL"], bins=PARAMS["BINS"])
    row["valley"] = float(valley) if np.isfinite(valley) else np.nan

    valley_for_init = float(valley) if metric == "turn_std" and np.isfinite(valley) else None
    _, gmm_stats = fit_gmm_2comp(x, valley_for_init, seed)

    row["pi_min"] = gmm_stats["pi_min"]
    row["cohen_d"] = gmm_stats["cohen_d"]
    row["delta_bic"] = compute_delta_bic(x, valley_for_init, seed)

    return row, valley


def run_gazebase_bimodality() -> pd.DataFrame:
    loader = GazeBaseLoader()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for game in tqdm(DEFAULT_GAMES, desc="games"):
        rounds = DEFAULT_ROUNDS_BY_GAME.get(game, [1])

        for metric in tqdm(METRICS, desc=f"metrics ({game})", leave=False):
            for ws in tqdm(WINDOWS, desc=f"windows ({game},{metric})", leave=False):
                stride = ws // 2
                seed = PARAMS["SEED"] + ws + (hash(game) & 0xFFFF) + (hash(metric) & 0xFFFF)

                # Pooled distribution used for bimodality analysis and plotting
                x, subjects = loader.pooled_metric(
                    game=game,
                    rounds=rounds,
                    metric=metric,
                    window_ms=ws,
                    stride_ms=stride,
                    max_subjects=MAX_SUBJECTS,
                    max_windows_per_subject=MAX_WINDOWS_PER_SUBJECT,
                    seed=seed,
                )
                x = x[np.isfinite(x)]
                n_subjects = len(subjects)

                # Subject effect diagnostic: variance explained by between subject mean differences
                per_subject = loader.per_subject_metric(
                    game=game,
                    rounds=rounds,
                    metric=metric,
                    window_ms=ws,
                    stride_ms=stride,
                    max_subjects=MAX_SUBJECTS,
                    max_windows_per_subject=MAX_WINDOWS_PER_SUBJECT,
                    seed=seed,
                )
                svr, n_subjects_for_svr = subject_variance_ratio_from_per_subject(x, per_subject)

                row, valley = analyze_metric_window(
                    x=x,
                    game=game,
                    rounds=rounds,
                    metric=metric,
                    ws=ws,
                    stride=stride,
                    seed=seed,
                    n_subjects=n_subjects,
                    subject_variance_ratio=svr,
                    n_subjects_for_svr=n_subjects_for_svr,
                )
                rows.append(row)

                out_dir = os.path.join(FIG_DIR, game)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{metric}_ws{ws}.pdf")
                make_plot(x, metric, out_path, valley=valley)

    df = pd.DataFrame(rows).sort_values(["game", "metric", "window_ms"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Figures: {FIG_DIR}")
    return df


if __name__ == "__main__":
    run_gazebase_bimodality()
