import os
import re
import time
import lightly_train
import torch
from pathlib import Path

TEST_FOLDS = [
    "Fold_1_Feb29_Mar11",
    "Fold_2_Mar01",
    "Fold_3_Mar05",
    "Fold_4_Mar09",
    "Fold_5_Mar10",
]
CV_ROOT = Path("/home/higo522/moose_deer/5_Fold_CV")

steps = 12000

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def main():
    import wandb  # ensure we can close runs between loop iterations

    for test_fold in TEST_FOLDS:
        heldout_root = CV_ROOT / test_fold / "CV" / "heldout_val"
        cv_splits = sorted([p for p in heldout_root.iterdir() if p.is_dir()])

        for split_path in cv_splits:
            # e.g. Fold_3_Mar05/CV/heldout_val/Fold1_val
            run_name = f"{test_fold}/CV/heldout_val/{split_path.name}"
            out_dir = f"Heldout_CV/{slugify(run_name)}"

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
                        "project": "LTDETR_nococo_rerun",
                        "name": run_name,
                        "log_model": False,
                    },
                    "val_every_num_steps": 500,
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
                    "optimizer_lr": 5e-5, 
                    "scheduler_warmup_steps": steps // 10,
                    "ema_warmup_steps": steps // 10,
                },
            )
            # Critical: close the run so next loop iteration starts a new one.
            wandb.finish()

if __name__ == "__main__":
    main()