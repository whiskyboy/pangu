"""Production ML scoring engine.

Discovers the latest walk-forward window models, computes Alpha158 factors
for a given date, and produces ensemble predictions.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pangu.factor.alpha158 import Alpha158Engine
from pangu.ml.model import LGBModel

if TYPE_CHECKING:
    from pangu.data.storage import Database

logger = logging.getLogger(__name__)

_MODEL_PATTERN = re.compile(r"wf_window_(\d+)_seed(\d+)\.txt")


class MLScorer:
    """Load model ensemble and score stocks.

    Discovers the latest walk-forward window from ``model_dir``, loads all
    seed models for that window, and averages predictions.

    Parameters
    ----------
    model_dir : directory containing ``wf_window_{NN}_seed{M}.txt`` files
    db : Database for loading bars / fundamentals
    """

    def __init__(self, model_dir: str, db: Database) -> None:
        self._model_dir = model_dir
        self._db = db
        self._engine = Alpha158Engine()
        self._models: list[LGBModel] = []
        self._window_id: int = -1
        self.reload()

    # ------------------------------------------------------------------
    # Model discovery & loading
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """(Re)discover and load models from model_dir."""
        model_path = Path(self._model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self._model_dir}")

        # Parse all model files
        entries: dict[int, list[Path]] = {}
        for f in sorted(model_path.glob("wf_window_*_seed*.txt")):
            m = _MODEL_PATTERN.match(f.name)
            if m:
                win_id = int(m.group(1))
                entries.setdefault(win_id, []).append(f)

        if not entries:
            raise FileNotFoundError(
                f"No model files matching wf_window_*_seed*.txt in {self._model_dir}"
            )

        # Use latest window
        latest_win = max(entries)
        paths = sorted(entries[latest_win])

        self._models = [LGBModel.load(str(p)) for p in paths]
        self._window_id = latest_win
        logger.info(
            "MLScorer loaded %d seed models from window %d (%s)",
            len(self._models), latest_win, self._model_dir,
        )

    @property
    def window_id(self) -> int:
        return self._window_id

    @property
    def n_models(self) -> int:
        return len(self._models)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, date: str, pool: list[str]) -> pd.Series:
        """Score all stocks in *pool* for a given *date*.

        Returns
        -------
        Series (index=symbol, name='score') — raw ensemble prediction.
        Higher is better.
        """
        if not self._models:
            raise RuntimeError("No models loaded. Call reload() or check model_dir.")

        factors = self._engine.compute_latest(self._db, date, pool)
        if factors.empty:
            logger.warning("MLScorer: compute_latest returned empty for %s", date)
            return pd.Series(dtype="float64", name="score")

        preds = [m.predict(factors) for m in self._models]
        avg = pd.concat(preds, axis=1).mean(axis=1)
        avg.name = "score"
        return avg
