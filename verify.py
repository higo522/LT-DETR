import torch
from pathlib import Path
from lightly_train._models.dinov3.dinov3_package import DINOv3Package
from lightly_train._task_models.task_model_helpers import download_checkpoint

BB = "ema_model.model.backbone.backbone."
ROOT = Path(__file__).parent

# ── Load all state dicts ───────────────────────────────────────────────────────
l1 = torch.load(ROOT / "hybrid_checkpoints/hybrid_coco_head_scratch_backbone.pt")["train_model"]
l2 = torch.load(ROOT / "hybrid_checkpoints/hybrid_coco_head_dinov3_backbone.pt")["train_model"]
l3 = torch.load(download_checkpoint("dinov3/convnext-small-ltdetr-coco"))["train_model"]
dinov3_sd = DINOv3Package.get_model("convnext-small").state_dict()

# ── Backbone samples (L1=random, L2=DINOv3, L3=COCO-trained DINOv3) ───────────
BB_KEYS = [BB + k for k in ["stages.0.0.gamma", "norm.weight"]]

print("=== BACKBONE (L2 should match DINOv3; L1 should differ from both) ===")
for k in BB_KEYS:
    print(f"\n  {k}  shape={l1[k].shape}")
    print(f"  L1 (scratch) : {l1[k][:5]}")
    print(f"  L2 (dinov3)  : {l2[k][:5]}")
    print(f"  L3 (coco)    : {l3[k][:5]}")
    print(f"  DINOv3 ref   : {dinov3_sd[k[len(BB):]][:5]}")
    print(f"  L2==DINOv3: {torch.allclose(l2[k], dinov3_sd[k[len(BB):]])}  |  L3==DINOv3: {torch.allclose(l3[k], dinov3_sd[k[len(BB):]])}  |  L1==DINOv3: {torch.allclose(l1[k], dinov3_sd[k[len(BB):]])}")

# ── Head samples (all three should be identical - same COCO weights) ──────────
HEAD_KEYS = [
    "ema_model.model.decoder.enc_bbox_head.layers.0.weight",
    "ema_model.model.decoder.decoder.layers.0.self_attn.in_proj_weight",
]

print("\n=== HEAD (L1==L2==L3 expected - all COCO) ===")
for k in HEAD_KEYS:
    print(f"\n  {k}  shape={l1[k].shape}")
    print(f"  L1 (scratch) : {l1[k][:5]}")
    print(f"  L2 (dinov3)  : {l2[k][:5]}")
    print(f"  L3 (coco)    : {l3[k][:5]}")
    print(f"  L1==L2: {torch.allclose(l1[k], l2[k])}  |  L1==L3: {torch.allclose(l1[k], l3[k])}  |  L2==L3: {torch.allclose(l2[k], l3[k])}")

'''
Backbone:
L2==DINOv3: True — the DINOv3 backbone was transplanted perfectly into the Level 2 hybrid
L3==DINOv3: False — Level 3's backbone was DINOv3 but then further trained on COCO, so the weights drifted slightly (e.g. norm.weight first value: 4.0846 → 4.1068)
L1==DINOv3: False — Level 1 has random init; gamma is all 1e-06 and norm.weight is all 1.0 which is what you'd expect from freshly initialized normalization layers

Head:
All three True across both head layers — the COCO-pretrained detection head is byte-for-byte identical in all three files
'''