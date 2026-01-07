import cv2
import albumentations as A
import os
import numpy as np

IMG_DIR = "data/processed/severity/train/images"
LBL_DIR = "data/processed/severity/train/labels"

# Augmentations (SAFE for corrosion)
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
    ],
    polygon_params=A.PolygonParams(format="yolo", label_fields=["class_labels"])
)

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.rsplit(".", 1)[0] + ".txt")

    if not os.path.exists(lbl_path):
        continue

    image = cv2.imread(img_path)
    h, w, _ = image.shape

    polygons = []
    labels = []

    # Read YOLO-seg labels
    with open(lbl_path, "r") as f:
        for line in f:
            parts = list(map(float, line.split()))
            cls = int(parts[0])
            coords = parts[1:]

            # reshape into (N,2)
            polygon = np.array(coords).reshape(-1, 2)
            polygons.append(polygon.tolist())
            labels.append(cls)

    # Create 2 augmented copies per image (3× total)
    for i in range(2):
        augmented = transform(
            image=image,
            polygons=polygons,
            class_labels=labels
        )

        aug_img = augmented["image"]
        aug_polygons = augmented["polygons"]
        aug_labels = augmented["class_labels"]

        new_name = img_name.rsplit(".", 1)[0] + f"_aug{i}.jpg"
        cv2.imwrite(os.path.join(IMG_DIR, new_name), aug_img)

        # Write new YOLO-seg label
        with open(os.path.join(LBL_DIR, new_name.replace(".jpg", ".txt")), "w") as f:
            for cls, poly in zip(aug_labels, aug_polygons):
                flat = " ".join(str(coord) for point in poly for coord in point)
                f.write(f"{cls} {flat}\n")
