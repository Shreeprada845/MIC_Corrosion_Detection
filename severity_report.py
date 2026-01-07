import os

LABEL_DIR = "runs/segment/predict3/labels"

severity_map = {
    0: "Low severity",
    1: "Moderate severity",
    2: "High severity",
    3: "Very high severity"
}

CONF_THRESHOLD = 0.1  # ignore weak predictions

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    highest_class = None

    with open(os.path.join(LABEL_DIR, file)) as f:
        for line in f:
            parts = line.split()
            cls = int(parts[0])
            conf = float(parts[1])

            if conf < CONF_THRESHOLD:
                continue

            if highest_class is None or cls > highest_class:
                highest_class = cls

    image_name = file.replace(".txt", "")

    if highest_class is None:
        print(f"{image_name} → Low severity")
    else:
        print(f"{image_name} → {severity_map[highest_class]}")
