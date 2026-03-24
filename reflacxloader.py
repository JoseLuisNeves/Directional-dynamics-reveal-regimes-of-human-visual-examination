import os, json
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Optional, Any
from itertools import islice
import numpy as np
import pandas as pd
import torch
from paths import dataset_paths
from local_annotations import AbnormalityEllipse, AnatomicalRegion
from gazebuilder import GazeWindow, build_gaze_windows
from utils import compute_scale_from_dims, scale_xy, scale_rect, filter_xy_to_rect
class ReflacxLoader:
    def __init__(self, root: Optional[str] = None, cache_dir: str = "outputs/cache/reflacx_windows"):
        self.root = Path(root or dataset_paths["reflacx"])
        self.cache_dir = cache_dir
        self.transcripts_dict, self.word_timestamps_dict, self.ellipses_dict, self.img_dims_dict, self.chest_dict = {}, {}, {}, {}, {}
        self._curr_dims_cache: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._scale_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}
    def load_jsons(self) -> None:
        file_map = {"transcripts.json": "transcripts_dict", "timestamps.json": "word_timestamps_dict","abnormality_ellipses.json": "ellipses_dict","img_dims.json": "img_dims_dict","chest_bbs.json": "chest_dict"}
        json_dir = self.root / "jsons"
        for filename, attr in file_map.items():
            with open(json_dir / filename, "r") as f:
                setattr(self, attr, json.load(f))
    def iter_study_pairs(self, max_studies: Optional[int] = None) -> Iterator[Tuple[str, str]]:
        pairs = ((pid, sid) for pid, studies in self.transcripts_dict.items() for sid in studies)
        return islice(pairs, max_studies)
    def get_study_annotations(self, patient_id: str, study_id: str) -> Tuple[AnatomicalRegion, List[AbnormalityEllipse]]:
        chest_raw = self.chest_dict[patient_id][study_id][0]
        chest_coords = (int(chest_raw["xmin"]), int(chest_raw["ymin"]), int(chest_raw["xmax"]), int(chest_raw["ymax"]))
        chest = AnatomicalRegion(coords=chest_coords, label="chest")
        ellipses: List[AbnormalityEllipse] = []
        for e in self.ellipses_dict[patient_id][study_id]:
            coords = (int(e["xmin"]), int(e["ymin"]), int(e["xmax"]), int(e["ymax"]))
            conds = {k: str(v).lower() == "true" for k, v in e.items() if k not in ("xmin","ymin","xmax","ymax","certainty")}
            labels = [c.lower() for c, present in conds.items() if present]
            ellipses.append(AbnormalityEllipse(coords=coords, labels=labels))
        return chest, ellipses
    def get_study_windows(self, patient_id: str, study_id: str, window_size: int, stride: int, *, filter_to_chest: bool = True, period: Optional[str] = None) -> List[GazeWindow]:
        gaze_path = os.path.join(self.root, "gaze_data", study_id, "gaze.csv")
        df = pd.read_csv(gaze_path)
        df = df.dropna(subset=["x_position","y_position","timestamp_sample"])
        xy = df[["x_position","y_position"]].to_numpy(dtype=np.float32)
        t = df["timestamp_sample"].to_numpy(dtype=np.float64)
        word_ts = self.word_timestamps_dict[patient_id][study_id]
        start_reporting_time = float(word_ts[0]["timestamp_start_word"])
        if filter_to_chest:
            chest, _ = self.get_study_annotations(patient_id, study_id)
            xy, t = filter_xy_to_rect(xy, t, chest.coords)
        return build_gaze_windows(xy, t, window_size=window_size, stride=stride, start_reporting_timestamp=start_reporting_time, period = period)
    def _cache_path(self, patient_id: str, study_id: str, window_size: int, stride: int, period: Optional[str] = None) -> str:
        d = os.path.join(self.cache_dir, f"ws{window_size}_st{stride}")
        os.makedirs(d, exist_ok=True)
        if period is not None:
            d = os.path.join(d, period)
            os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{patient_id}__{study_id}.npz")
    def get_study_window_features(self, patient_id: str, study_id: str, window_size: int, stride: int, *, filter_to_chest: bool = True, period: Optional[str] = None, use_cache: bool = True) -> Dict[str, np.ndarray]:
        cache_file = self._cache_path(patient_id, study_id, window_size, stride, period=period) if use_cache else None
        if use_cache and cache_file and os.path.exists(cache_file):
            z = np.load(cache_file)
            return {k: z[k] for k in z.files}
        windows = self.get_study_windows(patient_id, study_id, window_size, stride, filter_to_chest=filter_to_chest, period=period)
        dlist = [w.compute_features() for w in windows]
        feats = {k: np.asarray([d[k] for d in dlist], dtype=np.float32) for k in dlist[0].keys()}
        if use_cache and cache_file:
            np.savez_compressed(cache_file, **feats)
        return feats