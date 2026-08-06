import os
from pathlib import Path

CV_ROOT = Path(os.environ["SCRATCH"]) / "moose_deer" / "5_Fold_CV"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TEST_FOLDS = [
    "Fold_1_Feb29_Mar11",
    "Fold_2_Mar01",
    "Fold_3_Mar05",
    "Fold_4_Mar09",
    "Fold_5_Mar10",
]


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS) if folder.is_dir() else 0


def main():
    grand_totals = set()

    for test_fold in TEST_FOLDS:
        test_size = count_images(CV_ROOT / test_fold / "images" / "val")
        heldout_root = CV_ROOT / test_fold / "CV" / "heldout_val"
        print(f"\n{test_fold}  (test set: {test_size} images)")

        fold_totals = set()
        for split_path in sorted(p for p in heldout_root.iterdir() if p.is_dir()):
            train = count_images(split_path / "images" / "train")
            val = count_images(split_path / "images" / "val")
            total = train + val + test_size
            fold_totals.add(total)
            grand_totals.add(total)
            print(f"  {split_path.name:12s} train={train:5d}  val={val:5d}  train+val+test={total}")

        status = "OK" if len(fold_totals) == 1 else f"MISMATCH {fold_totals}"
        print(f"  -> {status}")

    print(f"\nAcross all 5 test folds: {'all totals match' if len(grand_totals) == 1 else f'MISMATCH {grand_totals}'}")


if __name__ == "__main__":
    main()