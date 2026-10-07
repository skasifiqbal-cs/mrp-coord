from omegaconf import DictConfig

from src.core.init.initializer import (
    BaseInitializer,
    FixedInitializer,
    RandomHeadingInitializer,
    RandomInitializer,
)

__all__ = ["BaseInitializer", "FixedInitializer", "RandomInitializer",
           "RandomHeadingInitializer", "build_initializer"]


def build_initializer(cfg_init: DictConfig) -> BaseInitializer:
    t = cfg_init.type
    if t == "fixed":
        return FixedInitializer()
    if t == "random_heading":
        return RandomHeadingInitializer()
    if t == "random":
        return RandomInitializer(
            margin=float(cfg_init.get("margin", 0.5)),
            max_attempts=int(cfg_init.get("max_attempts", 200)),
        )
    raise ValueError(f"Unknown init type: {t!r}. Choose 'fixed', 'random_heading', or 'random'.")
