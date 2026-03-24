import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
@dataclass
class GazeWindow:
    coords: np.ndarray
    timestamps: np.ndarray
    @property
    def center(self) -> Tuple[float, float]:
        return tuple(np.mean(self.coords, axis=0))
    @property
    def displacements(self) -> np.ndarray:
        return np.diff(self.coords, axis=0)
    @property
    def _dt(self) -> np.ndarray:
        dt = np.diff(self.timestamps)
        return np.where(dt == 0, np.nan, dt)
    @property
    def velocity(self) -> np.ndarray:
        return np.linalg.norm(self.displacements, axis=1) / self._dt
    @property
    def acceleration(self) -> np.ndarray:
        v = self.velocity
        dt = np.diff(self.timestamps[1:])
        dt = np.where(dt == 0, np.nan, dt)
        return np.diff(v) / dt
    @property
    def step_length(self) -> np.ndarray:
        return np.linalg.norm(self.displacements, axis=1)
    @property
    def turn_angles(self) -> np.ndarray:
        v1 = self.displacements[:-1]
        v2 = self.displacements[1:]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        denom = n1 * n2
        valid = denom > 0
        cos_theta = np.zeros(len(denom), dtype=np.float64)
        cos_theta[valid] = np.sum(v1[valid] * v2[valid], axis=1) / denom[valid]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
    @property
    def spatial_dispersion(self) -> float:
        m = np.mean(self.coords, axis=0)
        return float(np.sqrt(np.mean(np.sum((self.coords - m) ** 2, axis=1))))
    def compute_features(self) -> dict:
        turns = self.turn_angles
        vel = self.velocity
        acc = self.acceleration
        step = self.step_length
        return {"turn_std": float(np.nanstd(turns)) if len(turns) else np.nan,"velocity_std": float(np.nanstd(vel)) if len(vel) else np.nan, "acceleration_std": float(np.nanstd(acc)) if len(acc) else np.nan, "step_length_std": float(np.nanstd(step)) if len(step) else np.nan, "spatial_dispersion": self.spatial_dispersion}

def build_gaze_windows(positions: np.ndarray, timestamps: np.ndarray, window_size: int, stride: int, start_reporting_timestamp: Optional[float]=None, period: Optional[str] = None) -> List[GazeWindow]:
    windows = []
    for i in range(0, len(positions) - window_size + 1, stride):
        t_start = timestamps[i]
        if period == "reporting" and t_start < start_reporting_timestamp:
            continue
        if period == "pre-reporting" and t_start >= start_reporting_timestamp:
            continue  
        windows.append(GazeWindow(coords=positions[i : i + window_size], timestamps=timestamps[i : i + window_size]))
    return windows
