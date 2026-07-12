import torch
from lightly_train._models.dinov3.dinov3_package import DINOv3Package, MODEL_NAME_TO_INFO, _maybe_download_weights
from lightly_train._task_models.task_model_helpers import download_checkpoint

coco = torch.load(download_checkpoint("dinov3/convnext-small-ltdetr-coco"),
                  weights_only=False, map_location="cpu")

print("Top-level keys:", list(coco.keys()))

keys = list(coco["train_model"].keys())
print(f"\ntrain_model has {len(keys)} entries:")
for k, v in coco["train_model"].items():
    print(f"  {k}: {v.shape}")


# ── DINOv3 backbone-only weights ──────────────────────────────────────────────
dinov3_sd = torch.load(_maybe_download_weights(MODEL_NAME_TO_INFO["convnext-small"]),
                       weights_only=True, map_location="cpu")
print("\n=== DINOv3 backbone keys ===")
print(f"\nDINOv3 backbone has {len(list(dinov3_sd.keys()))} entries:")
for k, v in dinov3_sd.items():
    print(f"  {k}: {v.shape}")

# ── Random backbone (same architecture, random weights) ───────────────────────
random_sd = DINOv3Package.get_model("convnext-small", load_weights=False).state_dict()
print("\n=== Random backbone keys ===")
print(f"\nRandom backbone has {len(list(random_sd.keys()))} entries:")
for k, v in random_sd.items():
    print(f"  {k}: {v.shape}")

# backbone has 344 layers, while the whole model has 811 layers