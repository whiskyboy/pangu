"""T7: Monthly model update — expanding window + time decay training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)


async def update_model(c: Components) -> None:
    """Monthly model update: train single expanding window with time decay."""
    try:
        await _update_model_impl(c)
    except Exception:  # noqa: BLE001
        logger.error("[T7] Model update failed", exc_info=True)
        await c.alert("[T7] 模型更新失败，请检查日志")


async def _update_model_impl(c: Components) -> None:
    """Inner implementation of model update."""
    import asyncio

    from pangu.config import get_settings

    settings = get_settings()
    ml_cfg = settings.ml
    if not ml_cfg.get("enabled", False):
        logger.info("[T7] ML not enabled, skipping model update")
        return

    model_dir = ml_cfg.get("model_dir", "models")
    n_seeds = ml_cfg.get("n_seeds", 5)
    td_halflife = ml_cfg.get("time_decay_halflife", 120)
    first_train_start = "2020-01-01"

    latest_bar = c.db.get_latest_bar_date()
    if latest_bar is None:
        await c.alert("[T7] 无 K 线数据，无法更新模型")
        return

    logger.info(
        "[T7] Starting model update: expanding + TD%d, %d seeds, data up to %s",
        td_halflife, n_seeds, latest_bar,
    )
    await c.alert(f"[T7] 开始更新模型: expanding + TD{td_halflife}, {n_seeds} seeds")

    # Run training in executor to avoid blocking event loop
    from pangu.ml.model import train_walk_forward

    storage = c.market._storage

    loop = asyncio.get_running_loop()
    try:
        score_test, score_val = await loop.run_in_executor(
            None,
            lambda: train_walk_forward(
                storage=storage,
                model_dir=model_dir,
                output_dir="data",
                label_horizon=5,
                train_months=18,
                first_train_start=first_train_start,
                last_test_end=latest_bar,
                expanding=True,
                time_decay_halflife=td_halflife,
                n_seeds=n_seeds,
            ),
        )
    except Exception:
        logger.error("[T7] Training failed", exc_info=True)
        await c.alert("[T7] 模型训练失败，保留旧模型")
        return

    # Hot-swap: reload scorer with new models
    if c.ml_strategy is not None:
        try:
            c.ml_strategy._scorer.reload()
            logger.info(
                "[T7] Model hot-swapped: window=%d, seeds=%d",
                c.ml_strategy._scorer.window_id,
                c.ml_strategy._scorer.n_models,
            )
        except Exception:  # noqa: BLE001
            logger.error("[T7] Model reload failed", exc_info=True)
            await c.alert("[T7] 新模型加载失败，请手动检查")
            return

    msg = (
        f"[T7] ✅ 模型更新完成\n"
        f"  数据截止: {latest_bar}\n"
        f"  Seeds: {n_seeds}\n"
        f"  Time decay: {td_halflife}d"
    )
    if not score_test.empty:
        msg += f"\n  Test: {score_test.shape[0]}天 × {score_test.shape[1]}只"
    if not score_val.empty:
        msg += f"\n  Val: {score_val.shape[0]}天 × {score_val.shape[1]}只"

    logger.info("[T7] Model update complete")
    await c.alert(msg)
