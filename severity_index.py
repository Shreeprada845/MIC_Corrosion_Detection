import os
import csv
from shapely.geometry import Polygon

LABEL_DIR = "runs/segment/predict3/labels"
OUT_CSV = "severity_report.csv"

# weights
wc, wa, wp = 0.5, 0.3, 0.2

def severity_label(si):
    if si < 0.25:
        return "Low"
    elif si < 0.50:
        return "Moderate"
    elif si < 0.75:
        return "High"
    else:
        return "Very High"

rows = []

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    max_si = 0
    max_cls = None

    with open(os.path.join(LABEL_DIR, file)) as f:
        for line in f:
            parts = list(map(float, line.split()))
            cls = int(parts[0])
            conf = parts[1]
            coords = parts[2:]

            poly = Polygon([
                (coords[i], coords[i+1])
                for i in range(0, len(coords), 2)
            ])

            area = poly.area  # normalized [0,1]
            C = cls / 3
            A = area
            P = conf

            si = wc*C + wa*A + wp*P

            if si > max_si:
                max_si = si
                max_cls = cls

    if max_cls is None:
        rows.append([file.replace(".txt", ""), 0.0, "No corrosion", -1])
    else:
        rows.append([
            file.replace(".txt", ""),
            round(max_si, 3),
            severity_label(max_si),
            max_cls
        ])

# write CSV
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "severity_index", "severity_label", "max_class"])
    writer.writerows(rows)

print(f"CSV saved as {OUT_CSV}")
