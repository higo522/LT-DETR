import lightly_train

def main():
    lightly_train.train_object_detection(
        out="initial_test",
        model="dinov3/vits16-ltdetr-coco",
        batch_size=8,
        steps=12000,
        accelerator="gpu",
        devices=1,
        data={
            "format": "yolo",
            "path": "/home/higo522/RMNP_Dataset",
            "train": "train/images",
            "val": "val/images",
            "names": {0: "moose"},
        },
        logger_args={
            "wandb": {
                "project": "rmnp-detection",
                "name": "baseline(rect)",
                "log_model": False,
            },
            "val_every_num_steps": 1000,
        },
        save_checkpoint_args={
            "save_last": False,
            "save_best": True,
        },
        transform_args={
            "image_size": (480, 1280),
            "scale_jitter": None, # only works for square images, so we disable it for our rectangular input
        },
    )

if __name__ == "__main__":
    main()