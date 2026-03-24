import io, os, glob, zipfile, random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import re
from utils import dva_to_pixels
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from paths import dataset_paths
from gazebuilder import build_gaze_windows, GazeWindow
GAME_ABBREVIATIONS = {"Balura_Game": "BLG", "Fixations": "FXS", "Horizontal_Saccades": "HSS", "Random_Saccades": "RAN", "Reading": "TEX", "Video_1": "VD1", "Video_2": "VD2"}
FEATURE_KEYS = ("turn_std", "velocity_std", "acceleration_std", "step_length_std", "spatial_dispersion")
@dataclass(frozen=True)
class SubjectSession:
    subject: str
    session: str
    zip_path: str
    csv_name: str
def _safe_npz_path(cache_dir: str, game_abbr: str, ws: int, st: int, subject: str, session: str) -> str:
    d = os.path.join(cache_dir, game_abbr, f"ws{ws}_st{st}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{subject}__{session}.npz")

class GazeBaseLoader:
    def __init__(self, root: Optional[str] = None, cache_dir: str = "outputs/cache/gazebase_windows"):
        self.root = root or dataset_paths["gazebase"]
        self.cache_dir = cache_dir
    def iter_subject_sessions(self, game: str, rounds: Sequence[int], *, sessions: Sequence[str] = ("S1", "S2"), max_subjects: Optional[int] = None, seed: Optional[int] = 0) -> List[SubjectSession]:
        zips: List[Path] = []
        root_path = Path(self.root)
        for r in rounds:
            round_dir = root_path / f"Round_{int(r)}"
            if round_dir.is_dir():
                zips.extend(sorted(round_dir.glob("Subject_*.zip")))
        if seed is not None:
            random.Random(seed).shuffle(zips)
        if max_subjects is not None:
            zips = zips[:max_subjects]
        game_abbr = GAME_ABBREVIATIONS.get(game, game)
        out: List[SubjectSession] = []
        for zp in zips:
            m = re.search(r"Subject_(\d+)", zp.stem)
            subj = m.group(1) if m else zp.stem.replace("Subject_", "")
            for sess in sessions:
                out.append(SubjectSession(subject=subj,session=sess,zip_path=str(zp),csv_name=f"S_{subj}_{sess}_{game_abbr}.csv"))
        return out
    def get_subject_windows(self, ss: SubjectSession, window_ms: int, stride_ms: int) -> List[GazeWindow]:
        with zipfile.ZipFile(ss.zip_path, "r") as zf:
            matches = [f for f in zf.namelist() if f.endswith(ss.csv_name)]
            with zf.open(matches[0]) as f:
                df = pd.read_csv(f)
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        xy = dva_to_pixels(x[valid], y[valid])
        t = df["n"].to_numpy(dtype=float)[valid] * 0.001
        return build_gaze_windows(xy, t, window_size=int(window_ms), stride=int(stride_ms))
    def get_subject_window_features(self, ss: SubjectSession, window_ms: int, stride_ms: int,*, use_cache: bool = True) -> Dict[str, np.ndarray]:
        game_abbr = ss.csv_name.split("_")[-1].split(".")[0]
        cpath = _safe_npz_path(self.cache_dir, game_abbr, int(window_ms), int(stride_ms), ss.subject, ss.session)
        if use_cache and os.path.exists(cpath):
            z = np.load(cpath)
            return {k: z[k] for k in z.files}
        wins = self.get_subject_windows(ss, int(window_ms), int(stride_ms))
        dlist = [w.compute_features() for w in wins]
        feats = {k: np.asarray([d[k] for d in dlist], dtype=np.float32) for k in FEATURE_KEYS}
        if use_cache: np.savez_compressed(cpath, **feats)
        return feats
    def pooled_metric(self, game: str, rounds: Sequence[int], metric: str,window_ms: int, stride_ms: int, *, max_subjects: Optional[int] = None,max_windows_per_subject: int = 50_000,seed: int = 0) -> Tuple[np.ndarray, List[str]]:
        ss_list = self.iter_subject_sessions(game, rounds, max_subjects=max_subjects, seed=seed)
        rng = np.random.default_rng(seed)
        by_subject: Dict[str, List[np.ndarray]] = {}
        for ss in tqdm(ss_list, desc=f"pool {metric}", leave=False):
            feats = self.get_subject_window_features(ss, window_ms, stride_ms, use_cache=True)
            x = feats.get(metric, np.array([], dtype=np.float32)).astype(np.float64, copy=False)
            x = x[np.isfinite(x)]
            if x.size:
                by_subject.setdefault(ss.subject, []).append(x)
        subjects = sorted(by_subject.keys())
        pooled: List[np.ndarray] = []
        for s in subjects:
            x = np.concatenate(by_subject[s])
            if x.size > max_windows_per_subject:
                x = x[rng.choice(x.size, size=max_windows_per_subject, replace=False)]
            pooled.append(x)
        return (np.concatenate(pooled) if pooled else np.array([], dtype=np.float64)), subjects
    def per_subject_metric(self, game: str, rounds: Sequence[int], metric: str, window_ms: int, stride_ms: int, *, max_subjects: Optional[int] = None, max_windows_per_subject: int = 50_000, seed: int = 0) -> Dict[str, np.ndarray]:
        ss_list = self.iter_subject_sessions(game, rounds, max_subjects=max_subjects, seed=seed)
        rng = np.random.default_rng(seed)
        by_subject: Dict[str, List[np.ndarray]] = {}
        for ss in tqdm(ss_list, desc=f"per-sub {metric}", leave=False):
            feats = self.get_subject_window_features(ss, window_ms, stride_ms, use_cache=True)
            x = feats.get(metric, np.array([], dtype=np.float32)).astype(np.float64, copy=False)
            x = x[np.isfinite(x)]
            if x.size:
                by_subject.setdefault(ss.subject, []).append(x)
        out: Dict[str, np.ndarray] = {}
        for s, parts in by_subject.items():
            x = np.concatenate(parts)
            if x.size > max_windows_per_subject:
                x = x[rng.choice(x.size, size=max_windows_per_subject, replace=False)]
            out[s] = x
        return out


