"""T5: Monthly model update — single-window production training (all history + time decay)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pangu.tasks._base import scheduled_task

if TYPE_CHECKING:
    from pangu.scheduler import Components

logger = logging.getLogger(__name__)


@scheduled_task("T5", "月度模型更新")
async def update_model(c: Components) -> None:
    """Monthly model update: single expanding window with time decay.

    Uses :func:`pangu.ml.model.train` (production single-window) — NOT
    ``train_walk_forward`` (which is for backtest research). Producing only
    one window cuts runtime from ~2.5h to ~15–20 minutes; ``MLScorer.reload()``
    picks the latest window by ``max(window_id)``.
    """
    import asyncio

    from pangu.config import get_settings

    settings = get_settings()
    ml_cfg = settings.ml
    if not ml_cfg.get("enabled", False):
        logger.info("[T5] ML not enabled, skipping model update")
        return

    model_dir = ml_cfg.get("model_dir", "models")
    n_seeds = ml_cfg.get("n_seeds", 5)
    td_halflife = ml_cfg.get("time_decay_halflife", 120)
    first_train_start = ml_cfg.get("first_train_start", "2020-01-01")
    val_months = ml_cfg.get("val_months", 3)

    latest_bar = c.db.get_latest_bar_date()
    if latest_bar is None:
        await c.alert("[T5] 无 K 线数据，无法更新模型")
        return

    logger.info(
        "[T5] Starting model update: single-window + TD%d, %d seeds, data up to %s",
        td_halflife,
        n_seeds,
        latest_bar,
    )
    await c.alert(f"[T5] 开始更新模型: single-window + TD{td_halflife}, {n_seeds} seeds")

    from pangu.ml.model import train

    storage = c.market._storage

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: train(
                storage=storage,
                model_dir=model_dir,
                first_train_start=first_train_start,
                val_months=val_months,
                time_decay_halflife=td_halflife,
                n_seeds=n_seeds,
            ),
        )
    except Exception:
        logger.error("[T5] Training failed", exc_info=True)
        await c.alert("[T5] 模型训练失败，保留旧模型")
        return

    # Hot-swap: reload scorer with new models, or build the strategy from
    # scratch if this was a first-time (cold-start) train.
    if c.ml_strategy is not None:
        try:
            c.ml_strategy._scorer.reload()
            logger.info(
                "[T5] Model hot-swapped: window=%d, seeds=%d",
                c.ml_strategy._scorer.window_id,
                c.ml_strategy._scorer.n_models,
            )
        except Exception:  # noqa: BLE001
            logger.error("[T5] Model reload failed", exc_info=True)
            await c.alert("[T5] 新模型加载失败，请手动检查")
            return
    elif c.ml_enabled:
        # Cold-start path: ml_enabled was True but no models existed at
        # boot. After training succeeded, build the strategy in-place so
        # the next T6 can run without a scheduler restart.
        from pangu.strategy.ml.ml_strategy import try_build_ml_strategy

        new_strategy = try_build_ml_strategy(c.db, ml_cfg, settings.strategy)
        if new_strategy is None:
            logger.error("[T5] Training reported success but no models found in %s", model_dir)
            await c.alert("[T5] 训练完成但找不到模型文件，请人工检查")
            return
        c.ml_strategy = new_strategy
        logger.info(
            "[T5] Cold-start hot-build: window=%d, seeds=%d",
            c.ml_strategy._scorer.window_id,
            c.ml_strategy._scorer.n_models,
        )

    msg = (
        f"[T5] ✅ 模型更新完成\n"
        f"  数据截止: {latest_bar}\n"
        f"  Window:   {result['window_id']}\n"
        f"  Train:    {result['train_start']} ~ {result['train_end']} "
        f"({result['n_samples_train']:,} samples)\n"
        f"  Val:      {result['val_start']} ~ {result['val_end']} "
        f"({result['n_samples_val']:,} samples)\n"
        f"  Val IC:   {result['val_ic_mean']:.4f} (RankIC {result['val_rank_ic_mean']:.4f})\n"
        f"  Seeds:    {n_seeds}, TD halflife: {td_halflife}d"
    )

    logger.info("[T5] Model update complete (window=%d)", result["window_id"])
    await c.alert(msg)
