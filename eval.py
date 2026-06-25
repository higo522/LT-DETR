import os
import cv2
import numpy as np
import supervision as sv
import torch
import lightly_train
from tqdm import tqdm
from supervision.metrics import MeanAveragePrecision

# SIZES
# MODELS
MODELS = [
    {
        "name": "Model 1",
        "checkpoint": "experiments/LTDETR_nococo_rerun/fold_5_mar10_cv_heldout_val_fold1_val/exported_models/exported_best.pt",
    },
    {
        "name": "Model 2",
        "checkpoint": "experiments/LTDETR_nococo_rerun/fold_5_mar10_cv_heldout_val_fold2_val/exported_models/exported_best.pt",
    },
    {
        "name": "Model 3",
        "checkpoint": "experiments/LTDETR_nococo_rerun/fold_5_mar10_cv_heldout_val_fold3_val/exported_models/exported_best.pt",
    },
    {
        "name": "Model 4",
        "checkpoint": "experiments/LTDETR_nococo_rerun/fold_5_mar10_cv_heldout_val_fold4_val/exported_models/exported_best.pt",
    },
]

data_yaml_path = "/home/higo522/moose_deer/5_Fold_CV/test_data.yaml"
images_directory_path = "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10/images/val"
annotations_directory_path = "/home/higo522/moose_deer/5_Fold_CV/Fold_5_Mar10/labels/val"


for model_idx, model_config in enumerate(MODELS, start=1):
    checkpoint = model_config["checkpoint"]
    model_name = model_config["name"]

    print("/n" + "=" * 80)
    print(f"Model {model_idx}: {model_name}")
    print(f"Testing model checkpoint from {checkpoint}")
    print(f"Testing on dataset in {images_directory_path}")

    model = lightly_train.load_model(checkpoint)
    ds = sv.DetectionDataset.from_yolo(
        images_directory_path,
        annotations_directory_path,
        data_yaml_path
    )
    # annotations are in sv.Detections format (absolute pixels)

    targets = []
    predictions = []

    for path, image, annotations in tqdm(ds):
        # yields ("image name, image data, and corresponding annotation")
        raw_output = model.predict(path, threshold=0)  # absolute pixel xyxy
        detections = sv.Detections(
            xyxy=raw_output["bboxes"].cpu().numpy(),
            confidence=raw_output["scores"].cpu().numpy(),
            class_id=raw_output["labels"].cpu().numpy().astype(int)
        )

        targets.append(annotations)
        predictions.append(detections)

    cm = sv.ConfusionMatrix.from_detections(
        predictions=predictions,
        targets=targets,
        classes=['Moose', 'Deer'],
        conf_threshold=0.65,
        iou_threshold=0.5,
    ).matrix

    print(cm)

    TP_M = cm[0, 0]
    TP_D = cm[1, 1]

    FP_M = cm[:, 0].sum() - TP_M
    FP_D = cm[:, 1].sum() - TP_D

    FN_M = cm[0, :].sum() - TP_M
    FN_D = cm[1, :].sum() - TP_D

    precision_m = TP_M / (TP_M + FP_M + 1e-9)
    recall_m = TP_M / (TP_M + FN_M + 1e-9)
    F1_m = 2 * precision_m * recall_m / (precision_m + recall_m + 1e-9)

    precision_d = TP_D / (TP_D + FP_D + 1e-9)
    recall_d = TP_D / (TP_D + FN_D + 1e-9)
    F1_d = 2 * precision_d * recall_d / (precision_d + recall_d + 1e-9)

    macro_f1 = (F1_m + F1_d) / 2

    map_result = MeanAveragePrecision().update(predictions, targets).compute()

    print(f"MOOSE: Precision = {precision_m:.4f}, Recall = {recall_m:.4f}, F1 = {F1_m:.4f}, mAP50 = {map_result.ap_per_class[0,0]:.3f}, mAP50-95 = {map_result.ap_per_class[0].mean():.3f} ")
    print(f"DEER: Precision = {precision_d:.4f}, Recall = {recall_d:.4f}, F1 = {F1_d:.4f}, mAP50 = {map_result.ap_per_class[1,0]:.3f}, mAP50-95 = {map_result.ap_per_class[1].mean():.3f} ")
    print(f"Average: Precision = {(precision_m + precision_d) / 2:.4f}, Recall = {(recall_m + recall_d) / 2:.4f}, F1 = {macro_f1:.4f}, mAP50 = {map_result.ap_per_class[:,0].mean():.3f}, mAP50-95 = {map_result.ap_per_class.mean():.3f} ")
    