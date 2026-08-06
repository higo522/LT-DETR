import os
import re
import sys
from pathlib import Path

import lightly_train

TEST_FOLDS = [
    "Fold_1_Feb29_Mar11",
    "Fold_2_Mar01",
    "Fold_3_Mar05",
    "Fold_4_Mar09",
    "Fold_5_Mar10",
]
CV_ROOT = Path(os.environ["SCRATCH"]) / "moose_deer" / "5_Fold_CV"
BACKBONE_WEIGHTS = Path(os.environ["SCRATCH"]) / "LT-DETR" / "model_cache" / "dinov3_convnext_small_lvd1689m.pth"

steps = 72000


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def get_all_splits():
    # flatten (test_fold, split_path) into one list so SLURM_ARRAY_TASK_ID can index a single fold combo
    splits = []
    for test_fold in TEST_FOLDS:
        heldout_root = CV_ROOT / test_fold / "CV" / "heldout_val"
        for split_path in sorted(p for p in heldout_root.iterdir() if p.is_dir()):
            splits.append((test_fold, split_path))
    return splits


def main():
    import wandb

    task_id = int(sys.argv[1])
    test_fold, split_path = get_all_splits()[task_id]

    # e.g. Fold_3_Mar05/CV/heldout_val/Fold1_val
    run_name = f"{test_fold}/CV/heldout_val/{split_path.name}"
    out_dir = f"experiments/LTDETR_level_2_72k/{slugify(run_name)}"

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
                "project": "LTDETR_level_2_72k",
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
            # backbone_weights must be set explicitly: without it lightly_train forces
            # backbone_args.pretrained=False and randomly inits the backbone
            "backbone_weights": str(BACKBONE_WEIGHTS),
            "optimizer_lr": 1e-4,
            "scheduler_warmup_steps": steps // 10,
            "ema_warmup_steps": steps // 10,
        },
    )
    wandb.finish()


if __name__ == "__main__":
    main()
