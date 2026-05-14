import pandas as pd
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# DATASET ORGANIZATION
# ═════════════════════════════════════════════════════════════════════════════
def DatasetOrganization(dataset_path: str):
    records = []
    DATASET_DIR = Path(dataset_path)

    for patient_dir in DATASET_DIR.iterdir():
        if patient_dir.is_dir():

            patient_id = patient_dir.name

            for img_path in patient_dir.glob("*"):

                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:

                    records.append({
                        "patient_id": patient_id,
                        "image_path": str(img_path)
                    })

    return pd.DataFrame(records)