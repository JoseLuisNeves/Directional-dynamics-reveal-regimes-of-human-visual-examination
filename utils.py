from __future__ import annotations
import os
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import numpy as np
# Gazebase screen and viewing parameters
SCREEN_PX_W, SCREEN_PX_H = 1680, 1050
SCREEN_MM_W, SCREEN_MM_H = 474.0, 297.0
VIEWING_DISTANCE_MM = 550.0
EYE_VERTICAL_OFFSET_MM = 36.0
PIXEL_PITCH_X = SCREEN_MM_W / SCREEN_PX_W
PIXEL_PITCH_Y = SCREEN_MM_H / SCREEN_PX_H
def compute_scale_from_dims(orig_hw: Tuple[int, int], curr_hw: Tuple[int, int]) -> Tuple[float, float]:
    h0, w0 = orig_hw
    h1, w1 = curr_hw
    x_scale = float(w1) / float(max(1, w0))
    y_scale = float(h1) / float(max(1, h0))
    return x_scale, y_scale

def scale_xy(xy: np.ndarray, x_scale: float, y_scale: float, *, copy: bool = True) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float32)
    out = arr.copy() if copy else arr
    out[:, 0] *= x_scale
    out[:, 1] *= y_scale
    return out

def scale_rect(rect: Tuple[int, int, int, int], x_scale: float, y_scale: float) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = rect
    return (int(round(xmin * x_scale)),int(round(ymin * y_scale)),int(round(xmax * x_scale)),int(round(ymax * y_scale)))

def filter_xy_to_rect(xy: np.ndarray, t: Optional[np.ndarray], rect: Tuple[int, int, int, int]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    xmin, ymin, xmax, ymax = rect
    xy = np.asarray(xy)
    m = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
    xy_f = xy[m]
    if t is None:
        return xy_f, None
    t_f = np.asarray(t)[m]
    return xy_f, t_f

def dva_to_pixels(x_dva: np.ndarray, y_dva: np.ndarray) -> np.ndarray:
    x_dva = np.asarray(x_dva, dtype=float)
    y_dva = np.asarray(y_dva, dtype=float)
    mm_x = VIEWING_DISTANCE_MM * np.tan(np.radians(x_dva))
    mm_y = VIEWING_DISTANCE_MM * np.tan(np.radians(y_dva))
    px_x = mm_x / PIXEL_PITCH_X
    px_y = mm_y / PIXEL_PITCH_Y
    cx, cy = SCREEN_PX_W / 2.0, SCREEN_PX_H / 2.0
    eye_offset_px = EYE_VERTICAL_OFFSET_MM / PIXEL_PITCH_Y
    x_px = cx + px_x
    y_px = cy - px_y + eye_offset_px
    return np.vstack([x_px, y_px]).T.astype(np.float32, copy=False)

def savefig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
