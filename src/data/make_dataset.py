import pandas as pd
from glob import glob
import os

# 1. Configuration and Paths

data_path = "../../data/raw/Dataset(raw)"
subjects_info_path = "../../data/raw/data_subjects_info.csv"

LABEL_MAP = {
    'dws': 0, 'ups': 1, 'wlk': 2, 'jog': 3, 'sit': 4, 'std': 5
}

files = sorted(glob(data_path + "/*/*.csv"))

# 2. Read and Combine all files
all_df = pd.DataFrame()
data_set = 1

for f in files:
    folder_name = os.path.basename(os.path.dirname(f))
    activity = folder_name.split("_")[0]
    trial = folder_name.split("_")[1]
    user_id = int(os.path.basename(f).replace("sub_", "").replace(".csv", ""))

    df = pd.read_csv(f)

    df["user_id"] = user_id
    df["label"] = LABEL_MAP[activity]
    df["set"] = data_set

    all_df = pd.concat([all_df, df], ignore_index=True)
    data_set += 1

# 3. Integration: Merge with Subject Information

df_subjects = pd.read_csv(subjects_info_path)
df_subjects = df_subjects.rename(columns={'code': 'user_id'})

all_df = pd.merge(all_df, df_subjects, on='user_id', how='left')

# 4. Cleaning and Feature Construction

if "Unnamed: 0" in all_df.columns:
    del all_df["Unnamed: 0"]

all_df["acc_x"] = all_df["userAcceleration.x"] + all_df["gravity.x"]
all_df["acc_y"] = all_df["userAcceleration.y"] + all_df["gravity.y"]
all_df["acc_z"] = all_df["userAcceleration.z"] + all_df["gravity.z"]

all_df.rename(columns={
    'rotationRate.x': 'gyr_x',
    'rotationRate.y': 'gyr_y',
    'rotationRate.z': 'gyr_z'
}, inplace=True)

all_df.dropna(inplace=True)

all_df = all_df.drop(
    columns = ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z", "gravity.x", "gravity.y", "gravity.z"]
)
all_df = all_df[
    [
        "acc_x", "acc_y", "acc_z",
        "gyr_x", "gyr_y", "gyr_z",
        "attitude.roll", "attitude.pitch", "attitude.yaw",
        "user_id", "label", "set",
        "weight", "height", "age", "gender"
    ]
]

# 5. Working with Datetimes (50Hz = 20ms)

all_df["time_ms"] = all_df.groupby("set").cumcount() * 20
all_df.index = pd.to_datetime(all_df["time_ms"], unit="ms")

# 6. Resampling (Rule: 200ms)

sampling = {
    # Sensor
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "attitude.roll": "mean",
    "attitude.pitch": "mean",
    "attitude.yaw": "mean",
    # Metadata
    "label": "last",
    "user_id": "last",
    "set": "last",
    "weight": "last",
    "height": "last",
    "age": "last",
    "gender": "last"
}

data_resampled = (
    all_df
    .groupby(["user_id", "set"])
    .resample("200ms")
    .agg(sampling)
    .dropna()
)

data_resampled.reset_index(drop=True, inplace=True)

# 7. User-based Split (Train / Validation / Test)
users = sorted(data_resampled["user_id"].unique())

train_users = users[:18]
val_users = users[18:21]
test_users = users[21:]

train_df = data_resampled[data_resampled["user_id"].isin(train_users)]
val_df = data_resampled[data_resampled["user_id"].isin(val_users)]
test_df = data_resampled[data_resampled["user_id"].isin(test_users)]

assert set(train_users).isdisjoint(val_users)
assert set(train_users).isdisjoint(test_users)
assert set(val_users).isdisjoint(test_users)

# 8. Export Processed Data

output_dir = "../../data/processed/"
os.makedirs(output_dir, exist_ok=True)

data_resampled.to_pickle(os.path.join(output_dir, "data_processed.pkl"))
train_df.to_pickle(os.path.join(output_dir, "train.pkl"))
val_df.to_pickle(os.path.join(output_dir, "val.pkl"))
test_df.to_pickle(os.path.join(output_dir, "test.pkl"))