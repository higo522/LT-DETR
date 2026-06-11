import lightly_train
import wandb

LRS = [5e-5]
NAME="sweep_lr_detr_base"
steps = 30000

def train_with_config(config) -> None:
    run_name = (
        f"{config.lr:g}"
    )
    out_dir = f"sweep_lr/{NAME}/{run_name}-{wandb.run.id}"

    lightly_train.train_object_detection(
        out=out_dir,
        model="dinov3/convnext-base-ltdetr-coco",  # phase-2 finetune from this checkpoint (fold4_mar09)
        overwrite=True,
        batch_size=4,
        steps=steps,
        accelerator="gpu",
        devices=1,
        float32_matmul_precision="high",
        precision="bf16-mixed",  
        data={
            "path": "/home/higo522/moose_deer/5_Fold_CV/Fold_3_Mar05",
            "train": "images/train",
            "val": "images/val",
            "names": {0: "Moose", 1: "Deer"},
        },
        logger_args={
            "wandb": {
                "project": NAME,
                "name": run_name,
                "log_model": False,
            },
            "log_every_num_steps": 500,
            "val_every_num_steps": steps // 10,
        },
        save_checkpoint_args={
            "save_every_num_steps": steps // 5,    
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
            "optimizer_lr": float(config.lr),
            "scheduler_warmup_steps": steps // 10,
            "ema_warmup_steps": steps // 10,
        },
    )

def main():
    with wandb.init(project=NAME, reinit=True) as run:
        train_with_config(run.config)
    wandb.finish()

sweep_configuration = {
    "method": "grid",
    "parameters": {
        "lr": {"values": LRS},
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=NAME)
    wandb.agent(
        sweep_id,
        function=main,
        count=len(LRS) #* len(CONTEXT_WEIGHTS) #* len(CONTEXT_EXCLUSION_SIGMAS),  # NEW
    )