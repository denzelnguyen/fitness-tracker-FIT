import pandas as pd
from glob import glob
import os

# 1. Configuration and Paths

data_path = "../../data/raw/archive/A_DeviceMotion_data/A_DeviceMotion_data/"
subjects_info_path = "../../data/raw/archive/data_subjects_info.csv"

LABEL_MAP = {
    'dws': 0, 'ups': 1, 'wlk': 2, 'jog': 3, 'sit': 4, 'std': 5
}

files = glob(data_path + "*/*.csv")

# 2. Extract features from filename (Progressive Logic)

f = files[0]
folder_name = os.path.basename(os.path.dirname(f))
label = folder_name.split("_")[0]
category = folder_name.split("_")[1]
user_id = int(os.path.basename(f).replace("sub_", "").replace(".csv", ""))

# 3. Read and Combine all files

all_df = pd.DataFrame()
data_set = 1

for f in files:
    folder_name = os.path.basename(os.path.dirname(f))
    activity = folder_name.split("_")[0]
    trial = folder_name.split("_")[1]
    user_id = int(os.path.basename(f).replace("sub_", "").replace(".csv", ""))

    df = pd.read_csv(f)

    df["user_id"] = user_id
    df["label"] = LABEL_MAP.get(activity)
    df["category"] = activity
    df["set"] = data_set
    
    all_df = pd.concat([all_df, df], ignore_index=True)
    data_set += 1

# 4. Integration: Merge with Subject Information

df_subjects = pd.read_csv(subjects_info_path)
df_subjects = df_subjects.rename(columns={'code': 'user_id'})

all_df = pd.merge(all_df, df_subjects, on='user_id', how='left')

# 5. Cleaning and Renaming

if "Unnamed: 0" in all_df.columns:
    del all_df["Unnamed: 0"]

all_df.rename(columns={
    'userAcceleration.x': 'acc_x',
    'userAcceleration.y': 'acc_y',
    'userAcceleration.z': 'acc_z',
    'rotationRate.x': 'gyr_x',
    'rotationRate.y': 'gyr_y',
    'rotationRate.z': 'gyr_z'
}, inplace=True)

all_df.dropna(inplace=True)

# 6. Working with Datetimes (50Hz = 20ms)

all_df["time_ms"] = all_df.groupby("set").cumcount() * 20
all_df.index = pd.to_datetime(all_df["time_ms"], unit="ms")

# 7. Resampling (Rule: 200ms)

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",
    "category": "last",
    "user_id": "last",
    "set": "last",
    "weight": "last",
    "height": "last",
    "age": "last",
    "gender": "last"
}

data_resampled = all_df.groupby(["user_id", "set"]).resample("200ms").agg(sampling).dropna()
data_resampled.reset_index(drop=True, inplace=True)

# 8. Export Processed Data

output_path = "../../data/interim/01_data_processed.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
data_resampled.to_pickle(output_path)