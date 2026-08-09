import torch
from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
from pathlib import Path
import os

BACKBONE_WEIGHTS = Path(os.environ["SCRATCH"]) / "LT-DETR" / "model_cache" / "dinov3_convnext_small_lvd1689m.pth"

model = DINOV3_PACKAGE.get_model(
    "convnext-small",
    model_args={"pretrained": True, "weights": str(BACKBONE_WEIGHTS)},
)

cached = torch.load(str(BACKBONE_WEIGHTS), map_location="cpu")
# pick any key and compare
k = next(iter(cached))
print(torch.equal(model.state_dict()[k], cached[k]))  # True confirms it's loaded