"""Reproducible paired OE experiment; no production writes or test tuning."""

import argparse
import torch
from v4_phase2.pipeline import (
    protocol,
    train,
    extract,
    calibrate,
    freeze_evaluation,
    evaluate,
    BASE,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["fit", "evaluate", "all"], default="all")
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = protocol()
    runs = [
        f"{strategy}_seed{seed}"
        for strategy in ["random", "mined"]
        for seed in cfg["training_seeds"]
    ]
    if args.stage in ["fit", "all"]:
        for split in ["train", "val"]:
            extract("v3_recalibrated", split)
        calibrate("v3_recalibrated")
        for strategy in ["random", "mined"]:
            for seed in cfg["training_seeds"]:
                run = f"{strategy}_seed{seed}"
                train(strategy, seed)
                for split in ["train", "val"]:
                    extract(run, split)
                calibrate(run)
        freeze_evaluation()
    if args.stage in ["evaluate", "all"]:
        for run in ["v3_baseline", "v3_recalibrated"] + runs:
            extract(run, "test")
            evaluate(run)
        from v4_phase2.report import report

        report()


if __name__ == "__main__":
    main()
