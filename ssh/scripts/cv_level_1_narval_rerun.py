import os
import re
import sys
from pathlib import Path

import lightly_train

CV_ROOT = Path(os.environ["SCRATCH"]) / "moose_deer" / "5_Fold_CV"

# the two runs that failed in the full cv_level_1 sweep
RERUN_SPLITS = [
    ("Fold_1_Feb29_Mar11", "Fold2_val"),
    ("Fold_1_Feb29_Mar11", "Fold3_val"),
]

steps = 120000


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def main():
    import wandb

    task_id = int(sys.argv[1])
    test_fold, split_name = RERUN_SPLITS[task_id]
    split_path = CV_ROOT / test_fold / "CV" / "heldout_val" / split_name

    # e.g. Fold_1_Feb29_Mar11/CV/heldout_val/Fold2_val
    run_name = f"{test_fold}/CV/heldout_val/{split_path.name}"
    out_dir = f"experiments/LTDETR_level_1_120k/{slugify(run_name)}"

    lightly_train.train_object_detection(
        out=out_dir,
        model="dinov3/convnext-small-ltdetr",
        overwrite=True,
        batch_size=4,
        steps=steps,
        accelerator="gpu",
        devices=1,
        data={
            "path": str(split_path),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "Moose", 1: "Deer"},
        },
        logger_args={
            "wandb": {
                "project": "LTDETR_level_1_120k",
                "name": run_name,
                "log_model": False,
            },
            "val_every_num_steps": 1000,
        },
        save_checkpoint_args={
            "save_every_num_steps": 10000,
            "save_last": False,
            "save_best": True,
        },
        transform_args={
            "image_size": (640, 640),
            "random_flip": {"horizontal_prob": 0.5, "vertical_prob": 0.5},
            "photometric_distort": {
                "brightness": (0.875, 1.125),
                "contrast": (0.5, 1.5),
                "saturation": (1, 1),
                "hue": (0, 0),
                "prob": 0.5,
            },
        },
        model_args={
            "backbone_args": {"pretrained": False, "weights": None},
            "optimizer_lr": 1e-4,
            "backbone_lr": 1e-4,
            "scheduler_warmup_steps": steps // 10,
            "ema_warmup_steps": steps // 10,
        },
    )
    wandb.finish()


if __name__ == "__main__":
    main()