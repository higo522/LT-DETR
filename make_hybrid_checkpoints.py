# lightly_train 0.13.2 
# building a new train_model state_dict of the coco checkpoint with the COCO head and backbone portion swapped.
# backbone has 344 layers, while the whole model has 811 layers
# Top-level keys of coco: ['train_model', 'model_init_args', 'model_class_path']. The "train_model" key contains the actual state dict with the layer names and weight tensors. The other two keys are metadata about the model class and its initialization arguments.
# the exported COCO .pt contains exclusively EMA weights (# only keeps EMA weights for exporting). Thus only swapping the EMA stabilized backbone that was active at the end of COCO pretraining

import torch
from pathlib import Path
from lightly_train._models.dinov3.dinov3_package import DINOv3Package
from lightly_train._task_models.task_model_helpers import download_checkpoint

OUT = Path(__file__).parent / "hybrid_checkpoints"
OUT.mkdir(exist_ok=True)

# Every backbone weight in the COCO checkpoint sits behind this key prefix.
BB = "ema_model.model.backbone.backbone."

# ── Get the three source state dicts ──────────────────────────────────────────
coco     = torch.load(download_checkpoint("dinov3/convnext-small-ltdetr-coco"),
                      weights_only=False, map_location="cpu")
# download and read the train_model state dict from the COCO checkpoint (backbone + head weights), where the state dict for the backbone is nested under the key prefix "ema_model.model.backbone.model."
# coco["train_model"] is a dict containing all the actual layer names (keys) and their correspondiong weight tensors (values).
dinov3_sd = DINOv3Package.get_model("convnext-small").state_dict()
random_sd = DINOv3Package.get_model("convnext-small", load_weights=False).state_dict()
# these are dicts with the same backbone layer names but without the "BB" prefix
# ── Build and save each hybrid checkpoint ────────────────────────────────────
def make(backbone_sd, filename):
    new_sd = {
        k: backbone_sd[k[len(BB):]] if k.startswith(BB) else v
        for k, v in coco["train_model"].items()
    }
    # loops through every layer in coco["train_model"], gets the key (name of layer) and value (weight tensor) from the COCO train_model state dict
    # if the layer name starts with "BB" (backbone layer), strip the prefix BB off the name and grab the replacement tensor from backbone_sd. If not (head layer) keep the original COCO tensor untouched.
    # results in a new_sd with the same structure as coco["train_model"], same head weights as COCO, but backbone weights swapped out. 
    torch.save({**coco, "train_model": new_sd}, OUT / filename)
    # creates a copy of the full coco dict, but with train_model replaced by new_sd by spreading keys into new dict, and replace the "train_model" key with the new state dict that has the backbone portion swapped out for the provided backbone_sd.
    print(f"Saved {OUT / filename}")

make(random_sd, "hybrid_coco_head_scratch_backbone.pt")  # Level 1
make(dinov3_sd,  "hybrid_coco_head_dinov3_backbone.pt")   # Level 2

