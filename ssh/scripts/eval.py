import os
import sys
import csv
from pathlib import Path

import lightly_train
import supervision as sv
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision

CV_ROOT = Path(os.environ["SCRATCH"]) / "moose_deer" / "5_Fold_CV"
DATA_YAML = CV_ROOT / "test_data.yaml"

TEST_FOLDS = [
    {"name": "Test 1", "prefix": "fold_1_feb29_mar11", "data_dir": "Fold_1_Feb29_Mar11"},
    {"name": "Test 2", "prefix": "fold_2_mar01",       "data_dir": "Fold_2_Mar01"},
    {"name": "Test 3", "prefix": "fold_3_mar05",       "data_dir": "Fold_3_Mar05"},
    {"name": "Test 4", "prefix": "fold_4_mar09",       "data_dir": "Fold_4_Mar09"},
    {"name": "Test 5", "prefix": "fold_5_mar10",       "data_dir": "Fold_5_Mar10"},
]


def main():
    exp_name = "LTDETR_level_1"
    checkpoint_dir = Path("experiments") / exp_name
    csv_rows = []

    for test_fold in TEST_FOLDS:
        images_dir = CV_ROOT / test_fold["data_dir"] / "images" / "val"
        labels_dir = CV_ROOT / test_fold["data_dir"] / "labels" / "val"
        checkpoints = sorted(d for d in os.listdir(checkpoint_dir) if d.startswith(test_fold["prefix"]))

        for model_idx, ckpt in enumerate(checkpoints, start=1):
            checkpoint = checkpoint_dir / ckpt / "exported_models" / "exported_best.pt"
            model_name = f"Val {model_idx}"
            print(f"\n{test_fold['name']} — {model_name}: {checkpoint}")

            model = lightly_train.load_model(str(checkpoint))
            ds = sv.DetectionDataset.from_yolo(str(images_dir), str(labels_dir), str(DATA_YAML))

            targets, predictions = [], []
            for path, image, annotations in tqdm(ds):
                raw_output = model.predict(path, threshold=0)
                detections = sv.Detections(
                    xyxy=raw_output["bboxes"].cpu().numpy(),
                    confidence=raw_output["scores"].cpu().numpy(),
                    class_id=raw_output["labels"].cpu().numpy().astype(int),
                )
                targets.append(annotations)
                predictions.append(detections)

            cm = sv.ConfusionMatrix.from_detections(
                predictions=predictions,
                targets=targets,
                classes=["Moose", "Deer"],
                conf_threshold=0.65,
                iou_threshold=0.5,
            ).matrix

            TP_M, TP_D = cm[0, 0], cm[1, 1]
            FP_M, FP_D = cm[:, 0].sum() - TP_M, cm[:, 1].sum() - TP_D
            FN_M, FN_D = cm[0, :].sum() - TP_M, cm[1, :].sum() - TP_D
            precision_m, recall_m = TP_M / (TP_M + FP_M + 1e-9), TP_M / (TP_M + FN_M + 1e-9)
            precision_d, recall_d = TP_D / (TP_D + FP_D + 1e-9), TP_D / (TP_D + FN_D + 1e-9)
            F1_m = 2 * precision_m * recall_m / (precision_m + recall_m + 1e-9)
            F1_d = 2 * precision_d * recall_d / (precision_d + recall_d + 1e-9)
            macro_f1 = (F1_m + F1_d) / 2

            map_result = MeanAveragePrecision().update(predictions, targets).compute()

            csv_rows.append([test_fold["name"], model_name, "M", f"{precision_m:.4f}", f"{recall_m:.4f}", f"{F1_m:.4f}", f"{map_result.ap_per_class[0,0]:.3f}", f"{map_result.ap_per_class[0].mean():.3f}"])
            csv_rows.append([test_fold["name"], model_name, "D", f"{precision_d:.4f}", f"{recall_d:.4f}", f"{F1_d:.4f}", f"{map_result.ap_per_class[1,0]:.3f}", f"{map_result.ap_per_class[1].mean():.3f}"])
            csv_rows.append([test_fold["name"], model_name, "A", f"{(precision_m + precision_d) / 2:.4f}", f"{(recall_m + recall_d) / 2:.4f}", f"{macro_f1:.4f}", f"{map_result.ap_per_class[:,0].mean():.3f}", f"{map_result.ap_per_class.mean():.3f}"])

    csv_path = Path(os.environ["SCRATCH"]) / "LT-DETR" / f"results_{exp_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test Fold", "Val Model", "Class", "Precision", "Recall", "F1", "mAP50", "mAP50:95"])
        writer.writerows(csv_rows)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()