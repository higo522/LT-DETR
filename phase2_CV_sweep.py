import os
import re
import time

import lightly_train


# Phase-2 settings (match sweep.py)
OPTIMIZER_LR = 1e-5
CONTEXT_WEIGHTS = [1.0]  # add more values here, e.g. [0.0, 0.25, 0.5, 1.0, 2.0]
STEPS = 5000
BATCH_SIZE = 4

PROJECT = "phase2_cv_finetune(supervised_1)"
OUT_ROOT = "moose_deer_out(39)/phase2_cv(supervised_1)"


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


# IMPORTANT:
# Fill in the correct phase-1 checkpoint path for each fold.
# The dataset_path should match the fold being trained.
FOLDS = [
    {
        "dataset_path": "/home/higo522/moose_deer/5_Fold_CV/Fold_1_Feb29_Mar11",
        "phase1_checkpoint": "/home/higo522/lightly_train/runs/fold_1_feb29_mar11/exported_models/exported_last.pt",
    },
    {
        "dataset_path": "/home/higo522/moose_deer/5_Fold_CV/Fold_2_Mar01",
        "phase1_checkpoint": "/home/higo522/lightly_train/runs/fold_2_mar01/exported_models/exported_last.pt",
    },
    {
        "dataset_path": "/home/higo522/moose_deer/5_Fold_CV/Fold_3_Mar05",
        "phase1_checkpoint": "/home/higo522/lightly_train/runs/fold_3_mar05/exported_models/exported_last.pt",
    },
    {
        "dataset_path": "/home/higo522/moose_deer/5_Fold_CV/Fold_4_Mar09",
        "phase1_checkpoint": "/home/higo522/lightly_train/runs/fold_4_mar09/exported_models/exported_last.pt",
    },
    {
        "dataset_path": "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10",
        "phase1_checkpoint": "/home/higo522/lightly_train/runs/fold_5_mar10/exported_models/exported_last.pt",
    },
]


def main() -> None:
    import wandb  # local import so we can reliably finish runs per fold

    for context_weight in CONTEXT_WEIGHTS:
        for fold in FOLDS:
            dataset_path = fold["dataset_path"]
            ckpt = fold["phase1_checkpoint"]

            fold_id = os.path.basename(os.path.normpath(dataset_path))
            fold_slug = slugify(fold_id)

            run_name = f"{OPTIMIZER_LR:g}_ctx{context_weight:g}_{fold_slug}"
            out_dir = f"{OUT_ROOT}/{run_name}-{int(time.time())}"

            lightly_train.train_object_detection(
                out=out_dir,
                model=ckpt,  # phase-2 finetune starts from phase-1 exported_last.pt
                overwrite=True,
                batch_size=BATCH_SIZE,
                steps=STEPS,
                accelerator="gpu",
                devices=1,
                float32_matmul_precision="high",
                precision="bf16-mixed",
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
                    # Freeze backbone by setting its optimizer LR to 0
                    "backbone_lr": 0.0,
                    # Tune only this LR (applies to non-backbone param groups)
                    "optimizer_lr": float(OPTIMIZER_LR),
                    # Keep defaults / existing choices
                    "use_ema_model": True,
                    "matcher_use_focal_loss": True,
                    "scheduler_warmup_steps": 1000,
                    "ema_warmup_steps": 1000,
                    # Enable ONLY exclusion context loss (no count)
                    "criterion_losses": ["vfl", "boxes", "context_exclusion"],
                    "criterion_weight_dict": {
                        "loss_vfl": 1.0,
                        "loss_bbox": 5.0,
                        "loss_giou": 2.0,
                        "loss_context_exclusion": float(context_weight),
                    },
                    # Recommended for stability in phase-2 finetune
                    "criterion_context_start_step": 0,
                    "criterion_context_warmup_steps": 500,
                    "criterion_context_detach_boxes": True,
                },
            )

            # Close the W&B run so the next config starts a fresh run.
            wandb.finish()


if __name__ == "__main__":
    main()