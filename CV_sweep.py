import os
import re
import time
import lightly_train
import torch

DATASET_PATHS = [
    "/home/higo522/moose_deer/5_Fold_CV/Fold_2_Mar01",
    "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10",
    "/home/higo522/moose_deer/5_Fold_CV/Fold_3_Mar05",
    "/home/higo522/moose_deer/5_Fold_CV/Fold_4_Mar09",
    "/home/higo522/moose_deer/5_Fold_CV/Fold_1_Feb29_Mar11",
]

MODEL_NAME = "dinov3/convnext-base-ltdetr-coco"
PROJECT = "base_CV"


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def main():
    import wandb  # ensure we can close runs between loop iterations

    for dataset_path in DATASET_PATHS:
        dataset_id = os.path.basename(os.path.normpath(dataset_path))
        run_name = slugify(dataset_id)
        out_dir = f"runs/{run_name}"

        # Unique id so W&B cannot resume/merge runs unintentionally.
        run_id = f"{run_name}_{int(time.time())}"

        lightly_train.train_object_detection(
            out=out_dir,
            model=MODEL_NAME,
            overwrite=True,
            batch_size=4,
            steps=7500,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            float32_matmul_precision="high",
            data={
                "path": dataset_path,
                "train": "images/train",
                "val": "images/val",
                "names": {0: "Moose", 1: "Deer"},
            },
            logger_args={
                "wandb": {
                    "project": PROJECT,
                    "name": run_name,
                    "log_model": False,
                },
                "log_every_num_steps": 100,
                "val_every_num_steps": 2500,
            },
            save_checkpoint_args={"save_every_num_steps": 2500},
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
                "use_ema_model": True,
                "optimizer_lr": 1e-4,
                "matcher_use_focal_loss": True,
                "scheduler_warmup_steps": 2000,
                "ema_warmup_steps": 2000,
            },
        )

        # Critical: close the run so next loop iteration starts a new one.
        wandb.finish()


if __name__ == "__main__":
    main()

