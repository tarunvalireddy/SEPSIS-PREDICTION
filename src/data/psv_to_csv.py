import os
import glob
import pandas as pd

RAW_ROOT = "data/raw/"
OUTPUT_DIR = "data/raw/csv_real"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_psv_folder(folder):
    files = glob.glob(os.path.join(folder, "*.psv"))

    for file in files:
        df = pd.read_csv(file, sep="|")
        pid = os.path.basename(file).replace(".psv", "")
        df.to_csv(f"{OUTPUT_DIR}/{pid}.csv", index=False)

    print(f"✅ Converted {len(files)} files from {folder}")

if __name__ == "__main__":
    convert_psv_folder("data/raw/training_setA/training")
    convert_psv_folder("data/raw/training_setB/training")
    print("🎉 All PSV files converted to CSV")

