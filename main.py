"""Plan one scenario and score the plan in the simulator.

    python main.py approach.method=cegar env=open_cross_8_unicycle2
"""
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from src.approach import build_approach  # after sys.path is set by hydra
    build_approach(cfg).run(cfg)


if __name__ == "__main__":
    main()
