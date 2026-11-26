"""Look up the maps of past (training) wafers by index, for similar-wafer thumbnails."""
import json
from pathlib import Path

import numpy as np

from src.config import LABEL_MAP_PATH, PAST_WAFERS_PATH


class PastWaferStore:
    def __init__(self, path: Path = PAST_WAFERS_PATH, label_map_path: Path = LABEL_MAP_PATH):
        data = np.load(path)
        self.X, self.y = data["X_train"], data["y_train"]
        self.classes = json.loads(Path(label_map_path).read_text())

    def __len__(self) -> int:
        return len(self.X)

    def get(self, index: int) -> dict:
        if not 0 <= index < len(self.X):
            raise IndexError(index)
        return {"index": index,
                "label": self.classes[int(self.y[index])],
                "wafer_map": self.X[index].tolist()}