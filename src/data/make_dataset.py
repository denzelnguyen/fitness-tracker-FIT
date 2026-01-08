import pandas as pd
from glob import glob
import os

DATA_PATH = "../../data/raw/archive/A_DeviceMotion_data/A_DeviceMotion_data/"
SUBJECTS_INFO_PATH = "../../data/raw/archive/data_subjects_info.csv"

# Label mapping for activities
LABEL_MAP = {
    'dws': 0, 'ups': 1, 'wlk': 2, 'jog': 3, 'sit': 4, 'std': 5
}


def read_data_from_archive(data_path, subjects_info_path):
    df_subjects = pd.read_csv(subjects_info_path)
    df_subjects = df_subjects.rename(columns={'code': 'user_id'})

    files = glob(os.path.join(data_path, "*_*", "*.csv"))
    
    if not files:
        raise ValueError(f"No files found at {data_path}. Please check your directory structure.")

    all_chunks = []
    
    for f in files:
        folder_name = os.path.basename(os.path.dirname(f))
        file_name = os.path.basename(f)
        
        activity_code = folder_name.split('_')[0]
        trial_id = int(folder_name.split('_')[1])
        user_id = int(''.join(filter(str.isdigit, file_name)))

        df = pd.read_csv(f)
        
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

        # Labeling and Identification
        df["label"] = LABEL_MAP.get(activity_code)
        df["category"] = activity_code 
        df["user_id"] = user_id
        df["set"] = trial_id 
        
        all_chunks.append(df)

    full_df = pd.concat(all_chunks, ignore_index=True)

    full_df = pd.merge(full_df, df_subjects, on='user_id', how='left')

    return full_df


motion_df = read_data_from_archive(DATA_PATH, SUBJECTS_INFO_PATH)

initial_rows = len(motion_df)
motion_df.dropna(inplace=True)
if len(motion_df) < initial_rows:
    print(f"Dropped {initial_rows - len(motion_df)} rows containing NaN values.")

motion_df.rename(columns={
    'userAcceleration.x': 'acc_x',
    'userAcceleration.y': 'acc_y',
    'userAcceleration.z': 'acc_z',
    'rotationRate.x': 'gyr_x',
    'rotationRate.y': 'gyr_y',
    'rotationRate.z': 'gyr_z',
    'weight': 'weight',
    'height': 'height',
    'age': 'age',
    'gender': 'gender'
}, inplace=True)

motion_df = motion_df.sort_values(by=['user_id', 'set', 'label']).reset_index(drop=True)

output_path = "../../data/interim/01_data_processed.pkl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
motion_df.to_pickle(output_path)
