import lightly_train
import matplotlib.pyplot as plt
from torchvision import io, utils
import os

model = lightly_train.load_model("/home/higo522/lt-detr/moose_deer_out(39)/exclusion5_simple/exported_models/exported_last.pt")

input_dir = "/home/higo522/final dataset (12.1)/moose_deer/Date split/39d"
output_dir = "/home/higo522/lt-detr/moose_deer_out(39)/inference_test/39d/exclusion5_simple"

for root, _, files in os.walk(input_dir):
    rel_path = os.path.relpath(root, input_dir)
    out_dir = os.path.join(output_dir, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    for filename in files:
        img_path = os.path.join(root, filename)

        labels, boxes, scores = model.predict(img_path).values()
        image_with_boxes = utils.draw_bounding_boxes(
            image=io.read_image(img_path),
            boxes=boxes,
            labels=[f"{model.classes[i.item()]} {j.item():.2f}"
                    for i, j in zip(labels, scores)],
        )

        output_path = os.path.join(out_dir, filename)
        plt.imsave(
            output_path,
            image_with_boxes.permute(1, 2, 0).numpy()
        )
        print(f"Saved predicted image to {output_path}")
