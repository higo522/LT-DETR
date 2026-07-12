import os
import csv
import numpy as np
import supervision as sv
import lightly_train
from tqdm import tqdm

CHECKPOINT_DIR = "experiments/Heldout_CV (level 3)"
data_yaml_path = "/home/higo522/moose_deer/5_Fold_CV/test_data.yaml"

CONF_THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2).tolist()

TEST_FOLDS = [
    {"name": "Test 1", "prefix": "fold_1_feb29_mar11", "data_dir": "Fold_1_Feb29_Mar11"},
    {"name": "Test 2", "prefix": "fold_2_mar01",       "data_dir": "Fold_2_Mar01"},
    {"name": "Test 3", "prefix": "fold_3_mar05",       "data_dir": "Fold_3_Mar05"},
    {"name": "Test 4", "prefix": "fold_4_mar09",       "data_dir": "Fold_4_Mar09"},
    {"name": "Test 5", "prefix": "fold_5_mar10",       "data_dir": "Fold_5_Mar10"},
]

csv_rows = []

for test_fold in TEST_FOLDS:
    images_directory_path = f"/home/higo522/moose_deer/5_Fold_CV/{test_fold['data_dir']}/images/val"
    annotations_directory_path = f"/home/higo522/moose_deer/5_Fold_CV/{test_fold['data_dir']}/labels/val"
    checkpoints = sorted([d for d in os.listdir(CHECKPOINT_DIR) if d.startswith(test_fold["prefix"])])

    for ckpt in checkpoints:
        checkpoint = f"{CHECKPOINT_DIR}/{ckpt}/exported_models/exported_best.pt"
        model_name = ckpt[-9:]

        print("\n" + "=" * 80)
        print(f"{test_fold['name']} — {model_name}")
        print(f"Testing model checkpoint from {checkpoint}")
        print(f"Testing on dataset in {images_directory_path}")

        model = lightly_train.load_model(checkpoint)
        ds = sv.DetectionDataset.from_yolo(
            images_directory_path,
            annotations_directory_path,
            data_yaml_path
        )

        targets = []
        predictions = []

        for path, image, annotations in tqdm(ds):
            raw_output = model.predict(path, threshold=0)
            detections = sv.Detections(
                xyxy=raw_output["bboxes"].cpu().numpy(),
                confidence=raw_output["scores"].cpu().numpy(),
                class_id=raw_output["labels"].cpu().numpy().astype(int)
            )
            targets.append(annotations)
            predictions.append(detections)

        for conf in CONF_THRESHOLDS:
            cm = sv.ConfusionMatrix.from_detections(
                predictions=predictions,
                targets=targets,
                classes=['Moose', 'Deer'],
                conf_threshold=conf,
                iou_threshold=0.5,
            ).matrix

            TP_M = cm[0, 0]; TP_D = cm[1, 1]
            FP_M = cm[:, 0].sum() - TP_M; FP_D = cm[:, 1].sum() - TP_D
            FN_M = cm[0, :].sum() - TP_M; FN_D = cm[1, :].sum() - TP_D

            precision_m = TP_M / (TP_M + FP_M + 1e-9)
            recall_m    = TP_M / (TP_M + FN_M + 1e-9)
            F1_m        = 2 * precision_m * recall_m / (precision_m + recall_m + 1e-9)

            precision_d = TP_D / (TP_D + FP_D + 1e-9)
            recall_d    = TP_D / (TP_D + FN_D + 1e-9)
            F1_d        = 2 * precision_d * recall_d / (precision_d + recall_d + 1e-9)

            print(f"  conf={conf:.2f} | "
                  f"M: P={precision_m:.4f} R={recall_m:.4f} F1={F1_m:.4f} | "
                  f"D: P={precision_d:.4f} R={recall_d:.4f} F1={F1_d:.4f} | "
                  f"Avg: P={(precision_m + precision_d) / 2:.4f} R={(recall_m + recall_d) / 2:.4f} F1={(F1_m + F1_d) / 2:.4f}")

            csv_rows.append([test_fold["name"], model_name, f"{conf:.2f}", "M",
                             f"{precision_m:.4f}", f"{recall_m:.4f}", f"{F1_m:.4f}"])
            csv_rows.append([test_fold["name"], model_name, f"{conf:.2f}", "D",
                             f"{precision_d:.4f}", f"{recall_d:.4f}", f"{F1_d:.4f}"])
            csv_rows.append([test_fold["name"], model_name, f"{conf:.2f}", "A",
                             f"{(precision_m + precision_d) / 2:.4f}",
                             f"{(recall_m + recall_d) / 2:.4f}",
                             f"{(F1_m + F1_d) / 2:.4f}"])

csv_path = "conf_sweep.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Test Fold", "Val Model", "Conf Threshold", "Class", "Precision", "Recall", "F1"])
    writer.writerows(csv_rows)
print(f"\nResults saved to {csv_path}")
