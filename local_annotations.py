import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Union

@dataclass(frozen=True)
class AbnormalityEllipse:
    coords: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    labels: List[str]
    @property
    def center(self) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.coords
        return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    @property
    def axes(self) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.coords
        return (xmax - xmin) / 2.0, (ymax - ymin) / 2.0
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        cx, cy = self.center
        rx, ry = self.axes
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    def to_mask(self, height: int, width: int) -> np.ndarray:
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        return self.contains(xx, yy).astype(np.float32)

@dataclass(frozen=True)
class AnatomicalRegion:
    coords: Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
    label: str
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        xmin, ymin, xmax, ymax = self.coords
        return (xmin <= x) & (x <= xmax) & (ymin <= y) & (y <= ymax)
    def to_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.float32)
        xmin, ymin, xmax, ymax = self.coords
        mask[ymin:ymax + 1, xmin:xmax + 1] = 1.0
        return mask