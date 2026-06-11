import math
import supervision as sv
import lightly_train
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision

# --- Config ---
CHECKPOINT = "/home/higo522/lightly_train/runs/fold_5_mar10/exported_models/exported_last.pt"
DATA_YAML = "/home/higo522/moose_deer/5_Fold_CV/test_data.yaml"

DATASETS = [
    (
        "large",
        "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10/split_by_coco_size_val/large/images",
        "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10/split_by_coco_size_val/large/labels",
    ),
    (
        "medium",
        "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10/split_by_coco_size_val/medium/images",
        "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10/split_by_coco_size_val/medium/labels",
    ),
    (
        "small",
        "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10/split_by_coco_size_val/small/images",
        "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10/split_by_coco_size_val/small/labels",
    ),
]

# class indices assumed: 0=Moose, 1=Deer
MOOSE_ID = 0
DEER_ID = 1


def eval_dataset(model, images_dir: str, labels_dir: str, data_yaml: str):
    ds = sv.DetectionDataset.from_yolo(images_dir, labels_dir, data_yaml)

    preds, targs = [], []
    for path, _image, annotations in tqdm(ds, leave=False):
        out = model.predict(path, threshold=0)
        detections = sv.Detections(
            xyxy=out["bboxes"].cpu().numpy(),
            confidence=out["scores"].cpu().numpy(),
            class_id=out["labels"].cpu().numpy().astype(int),
        )
        preds.append(detections)
        targs.append(annotations)

    return MeanAveragePrecision().update(preds, targs).compute()


def safe_ap(map_result, class_id: int):
    ap = getattr(map_result, "ap_per_class", None)
    if ap is None or getattr(ap, "size", 0) == 0:
        return float("nan"), float("nan")
    if class_id < 0 or class_id >= ap.shape[0]:
        return float("nan"), float("nan")
    ap50 = float(ap[class_id, 0])
    ap5095 = float(ap[class_id].mean())
    return ap50, ap5095


def f(x: float) -> str:
    return "N/A" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"


def main():
    print(f"Checkpoint: {CHECKPOINT}")
    model = lightly_train.load_model(CHECKPOINT)

    for name, images_dir, labels_dir in DATASETS:
        print("\n" + "=" * 80)
        print(f"Dataset: {name}")
        print(f"Images : {images_dir}")
        print(f"Labels : {labels_dir}")

        r = eval_dataset(model, images_dir, labels_dir, DATA_YAML)

        moose_ap50, moose_ap5095 = safe_ap(r, MOOSE_ID)
        deer_ap50, deer_ap5095 = safe_ap(r, DEER_ID)

        print(f"mAP50:95 (all): {r.map50_95:.3f}")
        print(f"mAP50    (all): {r.map50:.3f}")
        print(f"Moose AP50: {f(moose_ap50)} | Moose AP50:95: {f(moose_ap5095)}")
        print(f"Deer  AP50: {f(deer_ap50)} | Deer  AP50:95: {f(deer_ap5095)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()