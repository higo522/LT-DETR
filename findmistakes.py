import json
from pathlib import Path
import lightly_train
import matplotlib.pyplot as plt
import torch
from torchvision import io, utils
from tqdm import tqdm
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
def _build_id_to_name(model) -> dict[int, str]:
    """
    Tries to build an id->name mapping from common Lightly/OD model attributes.
    Falls back to empty dict if unavailable.
    """
    # Common patterns
    for attr in ("id_to_name", "id2label", "class_names", "classes"):
        if not hasattr(model, attr):
            continue
        v = getattr(model, attr)
        if isinstance(v, dict):
            # assume {id: name}
            out: dict[int, str] = {}
            for k, name in v.items():
                try:
                    out[int(k)] = str(name)
                except Exception:
                    continue
            if out:
                return out
        if isinstance(v, (list, tuple)):
            # list[str] or list[int]
            if len(v) == 0:
                continue
            if all(isinstance(x, str) for x in v):
                return {i: n for i, n in enumerate(v)}
            if all(isinstance(x, int) for x in v):
                # ids only; no names
                return {int(i): str(int(i)) for i in v}
    return {}
def _find_class_id(
    id_to_name: dict[int, str],
    name: str,
    *,
    fallback_id: int | None = None,
) -> int:
    name_l = name.strip().lower()
    for cid, cname in id_to_name.items():
        if str(cname).strip().lower() == name_l:
            return int(cid)
    if fallback_id is not None:
        return int(fallback_id)
    raise ValueError(f"Class name '{name}' not found in id_to_name={id_to_name!r}")
def _iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p
def _guess_yolo_label_path(img_path: Path, images_root: Path) -> Path:
    """
    Best-effort mapping:
      .../images/val/xxx.jpg -> .../labels/val/xxx.txt
    If 'images' isn't in the path, it falls back to sibling 'labels'.
    """
    rel = img_path.relative_to(images_root)
    # Try replacing a 'images' path segment with 'labels' (common YOLO layout)
    parts = list(images_root.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        labels_root = Path(*parts)
    else:
        labels_root = images_root.parent / "labels"
    return (labels_root / rel).with_suffix(".txt")
def _load_yolo_gt(
    img_path: Path,
    images_root: Path,
    img_w: int,
    img_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      gt_labels: [G] int64
      gt_boxes_xyxy: [G,4] float32 in xyxy pixel coords
    If label file missing/unreadable -> returns empty tensors.
    """
    label_path = _guess_yolo_label_path(img_path, images_root)
    if not label_path.exists():
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
        )
    labels: list[int] = []
    boxes: list[list[float]] = []
    try:
        lines = label_path.read_text().strip().splitlines()
    except Exception:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
        )
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # YOLO: class cx cy w h (normalized)
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except Exception:
            continue
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
        boxes.append([x1, y1, x2, y2])
        labels.append(cid)
    if not labels:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
        )
    return (
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(boxes, dtype=torch.float32),
    )
@torch.no_grad()
def main():
    # ----------------- CONFIG (edit these) -----------------
    checkpoint = '/home/higo522/lightly_train/moose_deer_out(39)/Fold4_supervised_sweep2_2233n/ft_freeze_backbone_lr1e-05_ctx1-ux3q0woh/exported_models/exported_last.pt'
    #"/home/higo522/lightly_train/runs/fold_4_mar09/exported_models/exported_last.pt"
    #"/home/higo522/lt-detr/moose_deer_out(35)/final_sweep/ft_freeze_backbone_lr1e-05_ctx1_acc1-o14a3yy2/exported_models/exported_last.pt"
    input_dir = "/home/higo522/moose_deer/5_Fold_CV/Fold_4_Mar09/images/val"
    output_dir = "/home/higo522/lightly_train/phase2/fold4"
    threshold = 0.6
    # If your model doesn't store class names, set ids here:
    moose_id_fallback = 0
    deer_id_fallback = 1
    # If your model DOES store names, these will be used:
    moose_name = "moose"
    deer_name = "deer"
    copy_original = True
    # -------------------------------------------------------
    ckpt = Path(checkpoint)
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    # NEW: split categories
    out_dirs = {
        "multi_gt": out_root / "mixed_moose_deer_multi_gt",
        "single_gt": out_root / "mixed_moose_deer_single_gt_confusion",
    }
    for d in out_dirs.values():
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "annotated").mkdir(parents=True, exist_ok=True)
        (d / "meta").mkdir(parents=True, exist_ok=True)
    model = lightly_train.load_model(str(ckpt))
    id_to_name = _build_id_to_name(model)
    # If we only have numeric names, we likely don't have real class names stored
    only_numeric_names = all(v.isdigit() for v in id_to_name.values()) if id_to_name else True
    if only_numeric_names:
        moose_id = int(moose_id_fallback)
        deer_id = int(deer_id_fallback)
        # keep labels readable as ids
        name_for_id = lambda i: f"class_{int(i)}"
    else:
        moose_id = _find_class_id(id_to_name, moose_name, fallback_id=moose_id_fallback)
        deer_id = _find_class_id(id_to_name, deer_name, fallback_id=deer_id_fallback)
        name_for_id = lambda i: id_to_name.get(int(i), f"class_{int(i)}")
    all_paths = sorted(_iter_images(in_root))
    mistakes_multi: list[str] = []
    mistakes_single: list[str] = []
    for img_path in tqdm(all_paths, desc="Scanning images"):
        rel = img_path.relative_to(in_root)
        try:
            pred = model.predict(str(img_path), threshold=float(threshold))
        except TypeError:
            pred = model.predict(str(img_path))
        if not isinstance(pred, dict):
            raise RuntimeError(f"Expected model.predict to return dict, got {type(pred)}")
        # normalize key names across model variants
        if "boxes" not in pred and "bboxes" in pred:
            pred["boxes"] = pred["bboxes"]
        # robust key access
        if not all(k in pred for k in ("labels", "boxes", "scores")):
            raise RuntimeError(f"Prediction dict missing keys. Got keys={list(pred.keys())}")
        labels = pred["labels"]
        boxes = pred["boxes"]
        scores = pred["scores"]
        if labels.numel() == 0:
            continue
        has_moose = bool((labels == moose_id).any().item())
        has_deer = bool((labels == deer_id).any().item())
        if not (has_moose and has_deer):
            continue
        # --- NEW: load GT to decide category ---
        img = io.read_image(str(img_path))
        img_h = int(img.shape[1])
        img_w = int(img.shape[2])
        gt_labels, gt_boxes = _load_yolo_gt(img_path, in_root, img_w, img_h)
        gt_count = int(gt_labels.numel())
        category = "multi_gt" if gt_count >= 2 else "single_gt"
        base = out_dirs[category]
        dst_img_path = (base / "images") / rel
        dst_ann_path = (base / "annotated") / rel.with_suffix(".png")
        dst_meta_path = (base / "meta") / rel.with_suffix(".json")
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        dst_ann_path.parent.mkdir(parents=True, exist_ok=True)
        dst_meta_path.parent.mkdir(parents=True, exist_ok=True)
        if copy_original:
            dst_img_path.write_bytes(img_path.read_bytes())
        text_labels = [
            f"{name_for_id(int(c))} {float(s):.2f}"
            for c, s in zip(labels.cpu().tolist(), scores.cpu().tolist())
        ]
        annotated = utils.draw_bounding_boxes(
            image=img,
            boxes=boxes,
            labels=text_labels,
        )
        plt.imsave(str(dst_ann_path), annotated.permute(1, 2, 0).cpu().numpy())
        meta = {
            "image": str(rel),
            "checkpoint": str(ckpt),
            "threshold": float(threshold),
            "moose_id": int(moose_id),
            "deer_id": int(deer_id),
            "id_to_name": id_to_name,
            "gt": {
                "label_path_guess": str(_guess_yolo_label_path(img_path, in_root)),
                "num_boxes": gt_count,
                "labels": [int(x) for x in gt_labels.cpu().tolist()],
                "boxes_xyxy": [[float(v) for v in b] for b in gt_boxes.cpu().tolist()],
            },
            "predictions": [
                {
                    "class_id": int(c),
                    "class_name": name_for_id(int(c)),
                    "score": float(s),
                    "box_xyxy": [float(x) for x in b],
                }
                for c, s, b in zip(
                    labels.cpu().tolist(),
                    scores.cpu().tolist(),
                    boxes.cpu().tolist(),
                )
            ],
        }
        dst_meta_path.write_text(json.dumps(meta, indent=2))
        if category == "multi_gt":
            mistakes_multi.append(str(rel))
        else:
            mistakes_single.append(str(rel))
    (out_dirs["multi_gt"] / "index.json").write_text(json.dumps(mistakes_multi, indent=2))
    (out_dirs["single_gt"] / "index.json").write_text(json.dumps(mistakes_single, indent=2))
    print(f"Done.")
    print(f"Mixed moose+deer predictions with GT>=2: {len(mistakes_multi)} (saved to {out_dirs['multi_gt']})")
    print(f"Mixed moose+deer predictions with GT<=1: {len(mistakes_single)} (saved to {out_dirs['single_gt']})")
if __name__ == "__main__":
    main()
