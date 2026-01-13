import os
import numpy as np
import pandas as pd

OUTPUT_PATH = "data/raw/"
NUM_PATIENTS = 400
TIME_STEPS = 48
PREDICTION_HORIZON = 12

os.makedirs(OUTPUT_PATH, exist_ok=True)
np.random.seed(42)

def generate_patient(pid):
    hours = np.arange(TIME_STEPS)

    hr = np.random.normal(85, 15, TIME_STEPS)
    temp = np.random.normal(36.9, 0.6, TIME_STEPS)
    lactate = np.random.normal(1.5, 0.8, TIME_STEPS)
    sbp = np.random.normal(120, 20, TIME_STEPS)
    resp = np.random.normal(18, 5, TIME_STEPS)

    sepsis_onset = np.random.choice(
        [None] + list(range(22, 38)),
        p=[0.75] + [0.25 / 16] * 16
    )

    label = np.zeros(TIME_STEPS)

    if sepsis_onset:
        label[sepsis_onset:] = 1

        # subtle deterioration BEFORE onset
        start = max(sepsis_onset - 12, 0)
        hr[start:sepsis_onset] += np.linspace(3, 12, sepsis_onset - start)
        temp[start:sepsis_onset] += np.linspace(0.1, 0.8, sepsis_onset - start)
        lactate[start:sepsis_onset] += np.linspace(0.3, 1.2, sepsis_onset - start)

    df = pd.DataFrame({
        "Hour": hours,
        "HR": hr,
        "Temp": temp,
        "SBP": sbp,
        "Resp": resp,
        "Lactate": lactate,
        "SepsisLabel": label.astype(int)
    })

    df.to_csv(f"{OUTPUT_PATH}/p{pid:06d}.csv", index=False)

for i in range(1, NUM_PATIENTS + 1):
    generate_patient(i)

print("✅ HARD synthetic EHR data generated")

